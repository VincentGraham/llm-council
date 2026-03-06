"""N-round LLM council orchestration with structured evidence handling."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .config import (
    CHAIRMAN_MODEL,
    CONSENSUS_RATIO_THRESHOLD,
    COUNCIL_MODELS,
    EARLY_STOP_ENABLED_DEFAULT,
    EARLY_STOP_MIN_ROUNDS,
    ensure_runtime_config,
    ROUND1_INFERENCE_PARAMS,
    ROUND_N_INFERENCE_PARAMS,
    OBSERVER_CHAIRMAN_MODE,
    ROUNDS,
    SHARE_SYNTHESIS_WITH_MEMBERS,
    SYNTHESIS_INFERENCE_PARAMS,
    SYNTHESIS_SIMILARITY_THRESHOLD,
)
from .evidence import (
    build_evidence_index,
    mask_source_text,
    parse_hybrid_output,
    select_evidence_items,
)
from .inference import query_model, query_models_parallel
from .storage import (
    load_prompt_result,
    load_result,
    save_round,
    save_synthesis,
    update_prompt_result,
)

logger = logging.getLogger(__name__)
ROUND_N_CONTEXT_BUDGET_CHARS = max(1000, int(os.getenv("ROUND_N_CONTEXT_BUDGET_CHARS", "12000")))
BATCH_MAX_CONCURRENCY_DEFAULT = max(1, int(os.getenv("BATCH_MAX_CONCURRENCY", "2")))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Response {alphabet[index]}"
    return f"Response {index + 1}"


def _resolve_council_members() -> list[str]:
    """Resolve active deliberating members for a run."""
    members = list(COUNCIL_MODELS)
    if OBSERVER_CHAIRMAN_MODE:
        observer_members = [model for model in members if model != CHAIRMAN_MODEL]
        if observer_members:
            return observer_members
    return members


async def _save_round_async(**kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(save_round, **kwargs)


async def _save_synthesis_async(**kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(save_synthesis, **kwargs)


async def _update_prompt_result_async(**kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(update_prompt_result, **kwargs)


async def _load_result_async(batch_id: str) -> dict[str, Any] | None:
    return await asyncio.to_thread(load_result, batch_id)


async def _load_prompt_result_async(batch_id: str, prompt_id: str) -> dict[str, Any] | None:
    return await asyncio.to_thread(load_prompt_result, batch_id, prompt_id)


def _normalize_deliberation_input(
    prompt: str | None = None,
    deliberation_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = dict(deliberation_input or {})

    merged_prompt = source.get("prompt", prompt)
    trial_text = source.get("trial_text")
    if isinstance(merged_prompt, str):
        merged_prompt = merged_prompt.strip()
    if isinstance(trial_text, str):
        trial_text = trial_text.strip()

    inference = source.get("inference")
    if inference is not None and not isinstance(inference, dict):
        raise ValueError("'inference' must be an object keyed by stage name.")

    normalized = {
        "prompt": merged_prompt or None,
        "trial_text": trial_text or None,
        "prediction_target": source.get("prediction_target"),
        "allow_fuzzy_quotes": bool(source.get("allow_fuzzy_quotes", False)),
        "metadata": source.get("metadata"),
        "early_stopping": source.get("early_stopping"),
        "min_rounds_before_stop": source.get("min_rounds_before_stop"),
        "share_synthesis_with_members": source.get("share_synthesis_with_members"),
        "inference": inference,
    }

    if not normalized["prompt"] and not normalized["trial_text"]:
        raise ValueError("At least one of 'prompt' or 'trial_text' is required.")

    return normalized


def _primary_prompt_from_request(request_payload: dict[str, Any]) -> str:
    prompt = request_payload.get("prompt") or ""
    trial_text = request_payload.get("trial_text") or ""
    prediction_target = request_payload.get("prediction_target")

    if trial_text:
        duration_guidance = ""
        if isinstance(prediction_target, str) and "duration" in prediction_target.lower():
            duration_guidance = (
                "Target-specific guidance: predict the TOTAL trial duration from study "
                "initialization/start to completion of the primary endpoint "
                "(primary completion date). Do not predict intervention/treatment "
                "duration or follow-up-only windows unless they represent the full trial."
            )
        target_line = (
            f"Prediction target: {prediction_target}.\n"
            if prediction_target
            else "Prediction target: infer the most relevant requested clinical-trial outcome.\n"
        )
        preamble = """You are an expert forecaster for clinical trial outcomes.
Use the provided trial text to predict the requested target and explain your reasoning."""
        guidance_block = f"{duration_guidance}\n\n" if duration_guidance else ""
        base = f"{preamble}\n\n{target_line}{guidance_block}Clinical trial text:\n{trial_text}".strip()
        if prompt:
            return f"Additional user instructions:\n{prompt}\n\n{base}".strip()
        return base

    return prompt


def _source_text_for_anchoring(request_payload: dict[str, Any]) -> str:
    return request_payload.get("trial_text") or request_payload.get("prompt") or ""


STRUCTURED_OUTPUT_INSTRUCTIONS = """After your narrative answer, append a JSON object in a fenced ```json block.
The JSON must contain:
- prediction: {task_type, value_text, value_numeric, unit, probability, label, confidence}
- evidence: array of 3 to 5 items
Each evidence item must include: {quote, rationale, confidence, source_tag}
Use quote values copied from the source text whenever possible."""


ROUND_N_RANKING_INSTRUCTIONS = """Also include a FINAL RANKING section in your narrative before the JSON block:
FINAL RANKING:
1. Response X
2. Response Y
...
Only use provided labels."""


COUNCIL_MEMBER_SYSTEM_PROMPT = """You are an expert research council member.
Reason carefully before answering and use chain-of-thought style analysis internally.
In visible output, provide concise reasoning steps grounded in the provided text and avoid speculation."""


SYNTHESIS_SYSTEM_PROMPT = """You are the chairman of a research-grade LLM council.
Integrate evidence from all model rounds and produce a precise final synthesis with concise reasoning steps."""


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None

    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        try:
            normalized[key] = int(value)
        except (TypeError, ValueError):
            continue

    return normalized or None


def _sanitize_generation_params(stage_name: str, params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        logger.warning(
            "Ignoring inference override for stage '%s': expected object, got %s",
            stage_name,
            type(params).__name__,
        )
        return {}

    sanitized: dict[str, Any] = {}
    if params.get("temperature") is not None:
        try:
            temperature = float(params["temperature"])
            if 0.0 <= temperature <= 2.0:
                sanitized["temperature"] = temperature
            else:
                logger.warning(
                    "Ignoring invalid temperature override for stage '%s': %r",
                    stage_name,
                    params["temperature"],
                )
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric temperature override for stage '%s': %r",
                stage_name,
                params["temperature"],
            )
    if params.get("max_tokens") is not None:
        try:
            value = int(params["max_tokens"])
            if value > 0:
                sanitized["max_tokens"] = value
            else:
                logger.warning(
                    "Ignoring invalid max_tokens override for stage '%s': %r",
                    stage_name,
                    params["max_tokens"],
                )
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-integer max_tokens override for stage '%s': %r",
                stage_name,
                params["max_tokens"],
            )
    if params.get("top_p") is not None:
        try:
            top_p = float(params["top_p"])
            if 0.0 < top_p <= 1.0:
                sanitized["top_p"] = top_p
            else:
                logger.warning(
                    "Ignoring invalid top_p override for stage '%s': %r",
                    stage_name,
                    params["top_p"],
                )
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric top_p override for stage '%s': %r",
                stage_name,
                params["top_p"],
            )

    for key in params:
        if key not in {"temperature", "max_tokens", "top_p"}:
            logger.warning(
                "Ignoring unknown inference key for stage '%s': %s",
                stage_name,
                key,
            )
    return sanitized


def _resolve_stage_inference(
    request_payload: dict[str, Any],
    stage_name: str,
    default_params: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(default_params)
    request_inference = request_payload.get("inference")
    if not isinstance(request_inference, dict):
        return resolved

    stage_overrides = request_inference.get(stage_name)
    if stage_overrides is None and stage_name == "round_n":
        stage_overrides = request_inference.get("roundn")
    resolved.update(_sanitize_generation_params(stage_name, stage_overrides))
    return resolved


def _warn_unknown_inference_stages(request_payload: dict[str, Any]) -> None:
    request_inference = request_payload.get("inference")
    if not isinstance(request_inference, dict):
        return

    known = {"round1", "round_n", "roundn", "synthesis", "extractor"}
    for stage_name in request_inference:
        if stage_name not in known:
            logger.warning("Ignoring unknown inference stage override: %s", stage_name)


def _summarize_usage(
    all_rounds: list[dict[str, Any]],
    round_syntheses: list[dict[str, Any]],
) -> dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    calls_with_usage = 0

    def consume(model: str | None, usage: Any) -> None:
        nonlocal calls_with_usage
        normalized = _normalize_usage(usage)
        if normalized is None:
            return
        calls_with_usage += 1
        model_name = model or "unknown"
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = normalized.get(key, 0)
            totals[key] += value
            by_model[model_name][key] += value

    for round_data in all_rounds:
        for response in round_data.get("responses", []):
            consume(response.get("model"), response.get("usage"))

    for synthesis_entry in round_syntheses:
        synthesis = synthesis_entry.get("synthesis") or {}
        consume(synthesis.get("model"), synthesis.get("usage"))

    return {
        "calls_with_usage": calls_with_usage,
        "totals": totals,
        "by_model": dict(sorted(by_model.items(), key=lambda item: item[0])),
    }


def _format_responses_with_labels(
    responses: list[dict[str, Any]],
    exclude_model: str | None = None,
    label_by_model: dict[str, str] | None = None,
    model_order: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    lines = []
    mapping = {}

    response_by_model = {
        response.get("model"): response
        for response in responses
        if isinstance(response.get("model"), str)
    }
    ordered_models = model_order or [
        response.get("model")
        for response in responses
        if isinstance(response.get("model"), str)
    ]
    label_index = 0
    for model in ordered_models:
        if model == exclude_model:
            continue
        response = response_by_model.get(model)
        if not response or not response.get("response"):
            continue
        label = (label_by_model or {}).get(model) or _label_for_index(label_index)
        mapping[label] = model
        lines.append(f"{label}:\n{response['response']}")
        label_index += 1

    return lines, mapping


def _build_peer_label_map(council_members: list[str], model_name: str) -> dict[str, str]:
    peers = [member for member in council_members if member != model_name]
    return {peer: _label_for_index(index) for index, peer in enumerate(peers)}


def _truncate_prior_sections(prior_sections: list[str], max_chars: int) -> str:
    if not prior_sections:
        return "No prior responses available."
    if max_chars <= 0:
        return "\n\n".join(prior_sections)

    selected_reversed: list[str] = []
    used_chars = 0
    for section in reversed(prior_sections):
        section_chars = len(section) + 2
        if selected_reversed and (used_chars + section_chars) > max_chars:
            continue
        selected_reversed.append(section)
        used_chars += section_chars

    selected = list(reversed(selected_reversed))
    omitted_count = len(prior_sections) - len(selected)
    if omitted_count > 0:
        selected.insert(
            0,
            f"[{omitted_count} earlier round(s) omitted due to context budget of {max_chars} characters.]",
        )
    return "\n\n".join(selected)


def parse_ranking_from_text(ranking_text: str) -> list[str]:
    """Parse ranking labels from an LLM response."""
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:", maxsplit=1)
        ranking_section = parts[1] if len(parts) > 1 else ""
        numbered_matches = re.findall(r"\d+\.\s*Response [A-Z0-9]+", ranking_section)
        if numbered_matches:
            return [re.search(r"Response [A-Z0-9]+", m).group() for m in numbered_matches]
        matches = re.findall(r"Response [A-Z0-9]+", ranking_section)
        if matches:
            return matches
    return re.findall(r"Response [A-Z0-9]+", ranking_text)


def calculate_aggregate_rankings(
    stage_results: list[dict[str, Any]],
    label_to_model: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Calculate aggregate rankings from model reviews."""
    model_positions: dict[str, list[int]] = defaultdict(list)

    for result in stage_results:
        ranking_text = (
            result.get("ranking")
            or result.get("review")
            or result.get("raw_response")
            or result.get("response")
            or ""
        )
        parsed = result.get("parsed_ranking") or parse_ranking_from_text(ranking_text)
        mapping = label_to_model or result.get("label_to_model") or {}

        for position, label in enumerate(parsed, start=1):
            mapped_model = mapping.get(label)
            if mapped_model:
                model_positions[mapped_model].append(position)

    aggregate = []
    for model_name, positions in model_positions.items():
        if not positions:
            continue
        avg_rank = sum(positions) / len(positions)
        aggregate.append(
            {
                "model": model_name,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions),
            }
        )

    aggregate.sort(key=lambda item: item["average_rank"])
    return aggregate


def _prediction_signature(prediction: dict[str, Any] | None) -> str | None:
    if not isinstance(prediction, dict):
        return None
    parts: list[str] = []
    label = prediction.get("label")
    if label:
        parts.append(f"label:{str(label).strip().lower()}")
    probability = prediction.get("probability")
    if probability is not None:
        try:
            parts.append(f"prob:{round(float(probability), 3)}")
        except (TypeError, ValueError):
            pass
    numeric_value = prediction.get("value_numeric")
    if numeric_value is not None:
        try:
            unit = str(prediction.get("unit") or "").strip().lower()
            parts.append(f"num:{round(float(numeric_value), 3)}:{unit}")
        except (TypeError, ValueError):
            pass
    value_text = prediction.get("value_text")
    if value_text:
        parts.append(f"text:{str(value_text).strip().lower()}")
    if not parts:
        return None
    return "|".join(parts)


def _round_consensus_ratio(round_responses: list[dict[str, Any]]) -> float:
    signatures = []
    for response in round_responses:
        signature = _prediction_signature(response.get("prediction"))
        if signature:
            signatures.append(signature)

    if not signatures:
        return 0.0

    counts = defaultdict(int)
    for signature in signatures:
        counts[signature] += 1

    return max(counts.values()) / len(signatures)


def _synthesis_similarity(previous: str | None, current: str | None) -> float | None:
    if not previous or not current:
        return None
    prev_norm = " ".join(previous.split())
    curr_norm = " ".join(current.split())
    if not prev_norm or not curr_norm:
        return None
    return SequenceMatcher(None, prev_norm, curr_norm).ratio()


def _should_early_stop(
    round_number: int,
    early_stopping_enabled: bool,
    min_rounds_before_stop: int,
    consensus_ratio: float,
    synthesis_similarity: float | None,
) -> tuple[bool, str | None]:
    if not early_stopping_enabled:
        return False, None
    if round_number < min_rounds_before_stop:
        return False, None

    if consensus_ratio >= CONSENSUS_RATIO_THRESHOLD:
        return True, (
            f"consensus_ratio {consensus_ratio:.3f} >= threshold "
            f"{CONSENSUS_RATIO_THRESHOLD:.3f}"
        )

    if synthesis_similarity is not None and synthesis_similarity >= SYNTHESIS_SIMILARITY_THRESHOLD:
        return True, (
            f"synthesis_similarity {synthesis_similarity:.3f} >= threshold "
            f"{SYNTHESIS_SIMILARITY_THRESHOLD:.3f}"
        )

    return False, None


async def _parse_and_attach_structure(
    *,
    raw_text: str,
    request_payload: dict[str, Any],
    evidence_id_prefix: str,
) -> dict[str, Any]:
    parsed = await parse_hybrid_output(
        raw_text=raw_text,
        source_text=_source_text_for_anchoring(request_payload),
        allow_fuzzy_quotes=bool(request_payload.get("allow_fuzzy_quotes", False)),
        evidence_id_prefix=evidence_id_prefix,
    )
    return parsed


async def round_1(
    request_payload: dict[str, Any],
    council_members: list[str] | None = None,
    generation_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Round 1: all council models answer the prompt independently."""
    active_members = list(council_members or COUNCIL_MODELS)
    base_prompt = _primary_prompt_from_request(request_payload)
    messages = [
        {"role": "system", "content": COUNCIL_MEMBER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{base_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTIONS}",
        },
    ]
    responses = await query_models_parallel(
        active_members,
        messages,
        generation_params=generation_params,
    )

    round_responses = []
    for model in active_members:
        response = responses.get(model)
        if response is None:
            round_responses.append(
                {
                    "model": model,
                    "response": "",
                    "prediction": None,
                    "evidence": [],
                    "structured_json": None,
                    "structured_parse_status": "failed",
                    "structured_parse_errors": ["no_response_from_model"],
                    "usage": None,
                    "status": "error",
                    "error": "No response from model",
                }
            )
            continue

        raw_text = response.get("content", "")
        structured = await _parse_and_attach_structure(
            raw_text=raw_text,
            request_payload=request_payload,
            evidence_id_prefix=f"r1-{model}",
        )
        round_responses.append(
            {
                "model": model,
                "raw_response": raw_text,
                "response": structured["response"],
                "prediction": structured["prediction"],
                "evidence": structured["evidence"],
                "structured_json": structured["structured_json"],
                "structured_parse_status": structured["structured_parse_status"],
                "structured_parse_errors": structured["structured_parse_errors"],
                "reasoning_details": response.get("reasoning_details"),
                "usage": _normalize_usage(response.get("usage")),
                "status": "ok",
            }
        )

    return {
        "round": 1,
        "type": "initial",
        "created_at": _utcnow(),
        "responses": round_responses,
    }


def _build_round_n_prompt(
    model_name: str,
    request_payload: dict[str, Any],
    prior_rounds: list[dict[str, Any]],
    prior_syntheses: list[dict[str, Any]] | None,
    share_synthesis_with_members: bool,
    council_members: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    member_order = council_members
    if member_order is None:
        inferred_members: list[str] = []
        for prior_round in prior_rounds:
            for response in prior_round.get("responses", []):
                model = response.get("model")
                if isinstance(model, str) and model not in inferred_members:
                    inferred_members.append(model)
        member_order = inferred_members or COUNCIL_MODELS

    peer_order = [
        member
        for member in member_order
        if member != model_name
    ]
    peer_label_map = _build_peer_label_map(member_order, model_name)
    prior_sections = []

    for prior_round in prior_rounds:
        response_lines, _ = _format_responses_with_labels(
            prior_round["responses"],
            exclude_model=model_name,
            label_by_model=peer_label_map,
            model_order=peer_order,
        )
        if response_lines:
            prior_sections.append(
                f"Round {prior_round['round']} responses from OTHER models:\n\n"
                + "\n\n".join(response_lines)
            )

    latest_response_lines, latest_label_to_model = _format_responses_with_labels(
        prior_rounds[-1]["responses"],
        exclude_model=model_name,
        label_by_model=peer_label_map,
        model_order=peer_order,
    )

    prior_text = _truncate_prior_sections(prior_sections, ROUND_N_CONTEXT_BUDGET_CHARS)
    latest_text = (
        "\n\n".join(latest_response_lines) if latest_response_lines else "No prior responses available."
    )

    chairman_context = ""
    if share_synthesis_with_members and prior_syntheses:
        latest_synthesis = prior_syntheses[-1].get("synthesis", {}).get("response", "")
        if latest_synthesis:
            chairman_context = f"\n\nLatest chairman synthesis (for reference):\n{latest_synthesis}"

    base_prompt = _primary_prompt_from_request(request_payload)

    prompt_text = f"""You are participating in round {len(prior_rounds) + 1} of an LLM council.

Original task:
{base_prompt}

You are shown only anonymized responses from OTHER models in earlier rounds.

{prior_text}

For ranking, focus on the MOST RECENT prior round responses:
{latest_text}{chairman_context}

Your tasks:
1. Critique the other responses with concrete reasoning.
2. Provide your refined answer to the original task.
3. Include ranking and structured evidence output exactly as requested.

{ROUND_N_RANKING_INSTRUCTIONS}

{STRUCTURED_OUTPUT_INSTRUCTIONS}
"""

    return prompt_text, latest_label_to_model


async def round_n(
    request_payload: dict[str, Any],
    prior_rounds: list[dict[str, Any]],
    n: int,
    council_members: list[str] | None = None,
    prior_syntheses: list[dict[str, Any]] | None = None,
    share_synthesis_with_members: bool = False,
    generation_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Round n: each model critiques/refines using prior rounds from other models."""
    if n < 2:
        raise ValueError("round_n is only valid for rounds >= 2.")
    if not prior_rounds:
        raise ValueError("prior_rounds is required for round_n.")
    active_members = list(council_members or COUNCIL_MODELS)

    messages_by_model: dict[str, list[dict[str, str]]] = {}
    label_maps: dict[str, dict[str, str]] = {}
    for model in active_members:
        prompt_text, label_map = _build_round_n_prompt(
            model,
            request_payload,
            prior_rounds,
            prior_syntheses=prior_syntheses,
            share_synthesis_with_members=share_synthesis_with_members,
            council_members=active_members,
        )
        messages_by_model[model] = [{"role": "user", "content": prompt_text}]
        messages_by_model[model].insert(
            0,
            {"role": "system", "content": COUNCIL_MEMBER_SYSTEM_PROMPT},
        )
        label_maps[model] = label_map

    responses = await query_models_parallel(
        active_members,
        messages_by_model,
        generation_params=generation_params,
    )

    round_responses = []
    for model in active_members:
        response = responses.get(model)
        if response is None:
            round_responses.append(
                {
                    "model": model,
                    "response": "",
                    "review": "",
                    "prediction": None,
                    "evidence": [],
                    "structured_json": None,
                    "structured_parse_status": "failed",
                    "structured_parse_errors": ["no_response_from_model"],
                    "parsed_ranking": [],
                    "label_to_model": label_maps[model],
                    "usage": None,
                    "status": "error",
                    "error": "No response from model",
                }
            )
            continue

        raw_text = response.get("content", "")
        structured = await _parse_and_attach_structure(
            raw_text=raw_text,
            request_payload=request_payload,
            evidence_id_prefix=f"r{n}-{model}",
        )
        round_responses.append(
            {
                "model": model,
                "raw_response": raw_text,
                "response": structured["response"],
                "review": structured["response"],
                "prediction": structured["prediction"],
                "evidence": structured["evidence"],
                "structured_json": structured["structured_json"],
                "structured_parse_status": structured["structured_parse_status"],
                "structured_parse_errors": structured["structured_parse_errors"],
                "parsed_ranking": parse_ranking_from_text(raw_text),
                "label_to_model": label_maps[model],
                "reasoning_details": response.get("reasoning_details"),
                "usage": _normalize_usage(response.get("usage")),
                "status": "ok",
            }
        )

    aggregate_rankings = calculate_aggregate_rankings(round_responses)
    return {
        "round": n,
        "type": "deliberation",
        "created_at": _utcnow(),
        "responses": round_responses,
        "aggregate_rankings": aggregate_rankings,
    }


async def synthesize(
    request_payload: dict[str, Any],
    all_rounds: list[dict[str, Any]],
    synthesis_round: int | None = None,
    generation_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chairman synthesizes current council state."""
    rounds_text_parts = []
    for round_data in all_rounds:
        response_lines, _ = _format_responses_with_labels(round_data["responses"])
        rounds_text_parts.append(
            f"ROUND {round_data['round']} ({round_data.get('type', 'deliberation')}, anonymized):\n\n"
            + ("\n\n".join(response_lines) if response_lines else "No responses available.")
        )

    rounds_text = "\n\n".join(rounds_text_parts)
    base_prompt = _primary_prompt_from_request(request_payload)

    chairman_prompt = f"""You are the chairman of an LLM council.

Original task:
{base_prompt}

Council deliberation rounds seen so far:
{rounds_text}

Synthesize the best current answer for the user. Prioritize factual correctness,
clear reasoning, and actionable output when relevant.

{STRUCTURED_OUTPUT_INSTRUCTIONS}
"""

    response = await query_model(
        model=CHAIRMAN_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": chairman_prompt},
        ],
        **(generation_params or {}),
    )

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "synthesis_round": synthesis_round,
            "response": "Error: unable to generate synthesis.",
            "prediction": None,
            "evidence": [],
            "structured_json": None,
            "structured_parse_status": "failed",
            "structured_parse_errors": ["no_response_from_model"],
            "usage": None,
            "status": "error",
            "created_at": _utcnow(),
        }

    raw_text = response.get("content", "")
    round_suffix = f"r{synthesis_round}" if synthesis_round is not None else "final"
    structured = await _parse_and_attach_structure(
        raw_text=raw_text,
        request_payload=request_payload,
        evidence_id_prefix=f"s-{round_suffix}-{CHAIRMAN_MODEL}",
    )
    return {
        "model": CHAIRMAN_MODEL,
        "synthesis_round": synthesis_round,
        "raw_response": raw_text,
        "response": structured["response"],
        "prediction": structured["prediction"],
        "evidence": structured["evidence"],
        "structured_json": structured["structured_json"],
        "structured_parse_status": structured["structured_parse_status"],
        "structured_parse_errors": structured["structured_parse_errors"],
        "reasoning_details": response.get("reasoning_details"),
        "usage": _normalize_usage(response.get("usage")),
        "status": "ok",
        "created_at": _utcnow(),
    }


async def run_deliberation(
    prompt: str | None = None,
    rounds: int = ROUNDS,
    batch_id: str | None = None,
    prompt_id: str | None = None,
    prompt_index: int | None = None,
    prompt_count: int | None = None,
    deliberation_input: dict[str, Any] | None = None,
    counterfactual: dict[str, Any] | None = None,
    round_progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run dynamic deliberation trajectory and persist results."""
    ensure_runtime_config()
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not CHAIRMAN_MODEL:
        raise ValueError("No chairman model configured.")

    request_payload = _normalize_deliberation_input(prompt=prompt, deliberation_input=deliberation_input)
    display_prompt = request_payload.get("prompt") or request_payload.get("trial_text") or ""
    council_members = _resolve_council_members()
    if not council_members:
        raise ValueError("No active council members available for deliberation.")

    early_stopping_requested = request_payload.get("early_stopping")
    early_stopping_enabled = (
        EARLY_STOP_ENABLED_DEFAULT if early_stopping_requested is None else bool(early_stopping_requested)
    )
    share_synthesis_requested = request_payload.get("share_synthesis_with_members")
    share_synthesis_with_members = (
        SHARE_SYNTHESIS_WITH_MEMBERS
        if share_synthesis_requested is None
        else bool(share_synthesis_requested)
    )

    min_rounds_before_stop = request_payload.get("min_rounds_before_stop")
    if min_rounds_before_stop is None:
        min_rounds_before_stop = EARLY_STOP_MIN_ROUNDS
    try:
        min_rounds_before_stop = max(1, int(min_rounds_before_stop))
    except (TypeError, ValueError) as exc:
        raise ValueError("min_rounds_before_stop must be an integer >= 1") from exc

    _warn_unknown_inference_stages(request_payload)
    round1_params = _resolve_stage_inference(request_payload, "round1", ROUND1_INFERENCE_PARAMS)
    round_n_params = _resolve_stage_inference(request_payload, "round_n", ROUND_N_INFERENCE_PARAMS)
    synthesis_params = _resolve_stage_inference(request_payload, "synthesis", SYNTHESIS_INFERENCE_PARAMS)

    deliberation_meta = {
        "observer_chairman": OBSERVER_CHAIRMAN_MODE,
        "share_synthesis_with_members": share_synthesis_with_members,
        "early_stopping_enabled": early_stopping_enabled,
        "min_rounds_before_stop": min_rounds_before_stop,
        "synthesis_similarity_threshold": SYNTHESIS_SIMILARITY_THRESHOLD,
        "consensus_ratio_threshold": CONSENSUS_RATIO_THRESHOLD,
        "active_council_members": council_members,
        "chairman_model": CHAIRMAN_MODEL,
        "inference_params": {
            "round1": round1_params,
            "round_n": round_n_params,
            "synthesis": synthesis_params,
        },
    }

    resolved_batch_id = batch_id or str(uuid4())
    resolved_prompt_id = prompt_id or str(uuid4())

    all_rounds: list[dict[str, Any]] = []
    round_syntheses: list[dict[str, Any]] = []
    previous_synthesis_text: str | None = None
    early_stop_reason: str | None = None

    for round_number in range(1, rounds + 1):
        if round_number == 1:
            current_round = await round_1(
                request_payload,
                council_members=council_members,
                generation_params=round1_params,
            )
        else:
            current_round = await round_n(
                request_payload,
                all_rounds,
                round_number,
                council_members=council_members,
                prior_syntheses=round_syntheses,
                share_synthesis_with_members=share_synthesis_with_members,
                generation_params=round_n_params,
            )

        all_rounds.append(current_round)
        await _save_round_async(
            batch_id=resolved_batch_id,
            prompt_id=resolved_prompt_id,
            prompt=display_prompt,
            round_data=current_round,
            rounds_expected=rounds,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
            request_payload=request_payload,
            counterfactual=counterfactual,
        )

        observer_synthesis = await synthesize(
            request_payload=request_payload,
            all_rounds=all_rounds,
            synthesis_round=round_number,
            generation_params=synthesis_params,
        )
        consensus_ratio = _round_consensus_ratio(current_round.get("responses", []))
        similarity = _synthesis_similarity(previous_synthesis_text, observer_synthesis.get("response"))
        stop_triggered, stop_reason = _should_early_stop(
            round_number=round_number,
            early_stopping_enabled=early_stopping_enabled,
            min_rounds_before_stop=min_rounds_before_stop,
            consensus_ratio=consensus_ratio,
            synthesis_similarity=similarity,
        )

        round_syntheses.append(
            {
                "round": round_number,
                "synthesis": observer_synthesis,
                "consensus_ratio": round(consensus_ratio, 6),
                "synthesis_similarity_to_prev": (
                    round(similarity, 6) if similarity is not None else None
                ),
                "stop_triggered": stop_triggered,
                "stop_reason": stop_reason,
            }
        )

        await _update_prompt_result_async(
            batch_id=resolved_batch_id,
            prompt_id=resolved_prompt_id,
            updates={
                "round_syntheses": round_syntheses,
                "actual_rounds": len(all_rounds),
                "stopped_early": False,
                "early_stop_reason": None,
                "usage_summary": _summarize_usage(all_rounds, round_syntheses),
                "deliberation_meta": deliberation_meta,
            },
        )

        if round_progress_callback is not None:
            await round_progress_callback(
                {
                    "round": round_number,
                    "rounds_expected": rounds,
                    "actual_rounds_so_far": len(all_rounds),
                    "batch_id": resolved_batch_id,
                    "prompt_id": resolved_prompt_id,
                    "stopped_early": bool(stop_triggered),
                }
            )

        previous_synthesis_text = observer_synthesis.get("response")
        if stop_triggered:
            early_stop_reason = stop_reason
            break

    final_synthesis = (
        round_syntheses[-1]["synthesis"]
        if round_syntheses
        else await synthesize(
            request_payload,
            all_rounds,
            generation_params=synthesis_params,
        )
    )

    actual_rounds = len(all_rounds)
    stopped_early = actual_rounds < rounds

    stored = await _save_synthesis_async(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        prompt=display_prompt,
        synthesis=final_synthesis,
        rounds_expected=rounds,
        prompt_index=prompt_index,
        prompt_count=prompt_count,
        request_payload=request_payload,
        counterfactual=counterfactual,
    )

    stored["round_syntheses"] = round_syntheses
    evidence_index = build_evidence_index(stored)
    updated = await _update_prompt_result_async(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        updates={
            "evidence_index": evidence_index,
            "round_syntheses": round_syntheses,
            "actual_rounds": actual_rounds,
            "stopped_early": stopped_early,
            "early_stop_reason": early_stop_reason,
            "usage_summary": _summarize_usage(all_rounds, round_syntheses),
            "deliberation_meta": deliberation_meta,
            "schema_version": 3,
        },
    )

    return updated


async def run_batch_deliberation(
    prompts: list[str] | None = None,
    items: list[dict[str, Any]] | None = None,
    rounds: int = ROUNDS,
    batch_id: str | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Run deliberation for a batch of prompts/items with bounded concurrency."""
    has_prompts = bool(prompts)
    has_items = bool(items)
    if has_prompts == has_items:
        raise ValueError("Provide exactly one of 'prompts' or 'items'.")

    normalized_items: list[dict[str, Any]] = []
    if prompts:
        normalized_items = [{"prompt": prompt} for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
    else:
        normalized_items = [dict(item) for item in (items or [])]

    if not normalized_items:
        raise ValueError("No valid batch inputs provided.")

    resolved_batch_id = batch_id or str(uuid4())
    total_prompts = len(normalized_items)
    resolved_concurrency = max_concurrency
    if resolved_concurrency is None:
        resolved_concurrency = BATCH_MAX_CONCURRENCY_DEFAULT
    resolved_concurrency = max(1, min(int(resolved_concurrency), total_prompts))
    semaphore = asyncio.Semaphore(resolved_concurrency)

    async def run_item(index: int, item: dict[str, Any]) -> None:
        item_rounds = int(item.get("rounds") or rounds)
        async with semaphore:
            await run_deliberation(
                rounds=item_rounds,
                batch_id=resolved_batch_id,
                prompt_id=f"prompt-{index:04d}-{uuid4().hex[:8]}",
                prompt_index=index,
                prompt_count=total_prompts,
                deliberation_input=item,
            )

    await asyncio.gather(
        *(run_item(index, item) for index, item in enumerate(normalized_items, start=1))
    )

    stored_batch = await _load_result_async(resolved_batch_id) or {"batch": None, "results": []}
    return {
        "batch_id": resolved_batch_id,
        "rounds": rounds,
        "prompt_count": total_prompts,
        "max_concurrency": resolved_concurrency,
        "batch": stored_batch.get("batch"),
        "results": stored_batch.get("results", []),
    }


async def run_counterfactual_deliberation(
    source_batch_id: str,
    source_prompt_id: str,
    evidence_ids: list[str] | None = None,
    selectors: dict[str, Any] | None = None,
    rounds: int | None = None,
    allow_fuzzy_quotes: bool = False,
    metadata: dict[str, Any] | None = None,
    round_progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run a full counterfactual rerun with selected evidence masked out."""
    source_result = await _load_prompt_result_async(source_batch_id, source_prompt_id)
    if source_result is None:
        raise ValueError("Source prompt result not found.")

    evidence_index = source_result.get("evidence_index")
    if not evidence_index:
        evidence_index = build_evidence_index(source_result)

    selected_items = select_evidence_items(
        evidence_index=evidence_index,
        evidence_ids=evidence_ids,
        selectors=selectors,
    )

    source_request = dict(source_result.get("request") or {})
    base_source_text = source_request.get("trial_text") or source_request.get("prompt") or source_result.get("prompt") or ""

    masked_text, mask_manifest, masked_evidence_ids = mask_source_text(
        source_text=base_source_text,
        selected_items=selected_items,
    )
    if not masked_evidence_ids:
        raise ValueError("No maskable evidence items were selected.")

    counterfactual_request = {
        "prompt": source_request.get("prompt"),
        "trial_text": source_request.get("trial_text"),
        "prediction_target": source_request.get("prediction_target"),
        "allow_fuzzy_quotes": allow_fuzzy_quotes,
        "metadata": metadata or source_request.get("metadata"),
        "early_stopping": source_request.get("early_stopping"),
        "min_rounds_before_stop": source_request.get("min_rounds_before_stop"),
        "share_synthesis_with_members": source_request.get("share_synthesis_with_members"),
        "inference": source_request.get("inference"),
    }
    if counterfactual_request.get("trial_text"):
        counterfactual_request["trial_text"] = masked_text
    else:
        counterfactual_request["prompt"] = masked_text

    counterfactual_meta = {
        "parent_batch_id": source_batch_id,
        "parent_prompt_id": source_prompt_id,
        "masked_evidence_ids": masked_evidence_ids,
        "selectors": selectors,
        "mask_manifest": mask_manifest,
        "masked_input_preview": masked_text[:500],
    }

    rerun_rounds = int(rounds or source_result.get("rounds_expected") or ROUNDS)
    return await run_deliberation(
        rounds=rerun_rounds,
        deliberation_input=counterfactual_request,
        counterfactual=counterfactual_meta,
        round_progress_callback=round_progress_callback,
    )
