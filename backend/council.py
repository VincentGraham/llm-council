"""N-round LLM council orchestration with structured evidence handling."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, ROUNDS
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


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Response {alphabet[index]}"
    return f"Response {index + 1}"


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

    normalized = {
        "prompt": merged_prompt or None,
        "trial_text": trial_text or None,
        "prediction_target": source.get("prediction_target"),
        "allow_fuzzy_quotes": bool(source.get("allow_fuzzy_quotes", False)),
        "metadata": source.get("metadata"),
    }

    if not normalized["prompt"] and not normalized["trial_text"]:
        raise ValueError("At least one of 'prompt' or 'trial_text' is required.")

    return normalized


def _primary_prompt_from_request(request_payload: dict[str, Any]) -> str:
    prompt = request_payload.get("prompt") or ""
    trial_text = request_payload.get("trial_text") or ""
    prediction_target = request_payload.get("prediction_target")

    if trial_text:
        target_line = (
            f"Prediction target: {prediction_target}.\n"
            if prediction_target
            else "Prediction target: infer the most relevant requested clinical-trial outcome.\n"
        )
        preamble = """You are an expert forecaster for clinical trial outcomes.
Use the provided trial text to predict the requested target and explain your reasoning."""
        base = f"{preamble}\n\n{target_line}\nClinical trial text:\n{trial_text}".strip()
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


def _format_responses_with_labels(
    responses: list[dict[str, Any]],
    exclude_model: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    lines = []
    mapping = {}

    filtered = [
        response
        for response in responses
        if response.get("model") != exclude_model and response.get("response")
    ]
    for index, response in enumerate(filtered):
        label = _label_for_index(index)
        mapping[label] = response["model"]
        lines.append(f"{label}:\n{response['response']}")

    return lines, mapping


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


async def round_1(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Round 1: all council models answer the prompt independently."""
    base_prompt = _primary_prompt_from_request(request_payload)
    messages = [{
        "role": "user",
        "content": f"{base_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTIONS}",
    }]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    round_responses = []
    for model in COUNCIL_MODELS:
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
) -> tuple[str, dict[str, str]]:
    prior_sections = []

    for prior_round in prior_rounds:
        response_lines, _ = _format_responses_with_labels(
            prior_round["responses"], exclude_model=model_name
        )
        if response_lines:
            prior_sections.append(
                f"Round {prior_round['round']} responses from OTHER models:\n\n"
                + "\n\n".join(response_lines)
            )

    latest_response_lines, latest_label_to_model = _format_responses_with_labels(
        prior_rounds[-1]["responses"],
        exclude_model=model_name,
    )

    prior_text = "\n\n".join(prior_sections) if prior_sections else "No prior responses available."
    latest_text = (
        "\n\n".join(latest_response_lines) if latest_response_lines else "No prior responses available."
    )

    base_prompt = _primary_prompt_from_request(request_payload)

    prompt_text = f"""You are participating in round {len(prior_rounds) + 1} of an LLM council.

Original task:
{base_prompt}

You are shown only anonymized responses from OTHER models in earlier rounds.

{prior_text}

For ranking, focus on the MOST RECENT prior round responses:
{latest_text}

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
) -> dict[str, Any]:
    """Round n: each model critiques/refines using prior rounds from other models."""
    if n < 2:
        raise ValueError("round_n is only valid for rounds >= 2.")
    if not prior_rounds:
        raise ValueError("prior_rounds is required for round_n.")

    messages_by_model: dict[str, list[dict[str, str]]] = {}
    label_maps: dict[str, dict[str, str]] = {}
    for model in COUNCIL_MODELS:
        prompt_text, label_map = _build_round_n_prompt(model, request_payload, prior_rounds)
        messages_by_model[model] = [{"role": "user", "content": prompt_text}]
        label_maps[model] = label_map

    responses = await query_models_parallel(COUNCIL_MODELS, messages_by_model)

    round_responses = []
    for model in COUNCIL_MODELS:
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
) -> dict[str, Any]:
    """Chairman synthesizes final response from all rounds."""
    rounds_text_parts = []
    for round_data in all_rounds:
        response_lines = []
        for response in round_data["responses"]:
            response_lines.append(
                f"Model: {response.get('model')}\nResponse:\n{response.get('response', '')}"
            )
        rounds_text_parts.append(
            f"ROUND {round_data['round']} ({round_data.get('type', 'deliberation')}):\n\n"
            + "\n\n".join(response_lines)
        )

    rounds_text = "\n\n".join(rounds_text_parts)
    base_prompt = _primary_prompt_from_request(request_payload)

    chairman_prompt = f"""You are the chairman of an LLM council.

Original task:
{base_prompt}

Council deliberation rounds:
{rounds_text}

Synthesize a final answer for the user. Prioritize factual correctness,
clear reasoning, and actionable output when relevant.

{STRUCTURED_OUTPUT_INSTRUCTIONS}
"""

    response = await query_model(
        model=CHAIRMAN_MODEL,
        messages=[{"role": "user", "content": chairman_prompt}],
    )

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: unable to generate synthesis.",
            "prediction": None,
            "evidence": [],
            "structured_json": None,
            "structured_parse_status": "failed",
            "structured_parse_errors": ["no_response_from_model"],
            "status": "error",
            "created_at": _utcnow(),
        }

    raw_text = response.get("content", "")
    structured = await _parse_and_attach_structure(
        raw_text=raw_text,
        request_payload=request_payload,
        evidence_id_prefix=f"s-{CHAIRMAN_MODEL}",
    )
    return {
        "model": CHAIRMAN_MODEL,
        "raw_response": raw_text,
        "response": structured["response"],
        "prediction": structured["prediction"],
        "evidence": structured["evidence"],
        "structured_json": structured["structured_json"],
        "structured_parse_status": structured["structured_parse_status"],
        "structured_parse_errors": structured["structured_parse_errors"],
        "reasoning_details": response.get("reasoning_details"),
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
) -> dict[str, Any]:
    """Run full N-round deliberation and persist results."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    request_payload = _normalize_deliberation_input(prompt=prompt, deliberation_input=deliberation_input)
    display_prompt = request_payload.get("prompt") or request_payload.get("trial_text") or ""

    resolved_batch_id = batch_id or str(uuid4())
    resolved_prompt_id = prompt_id or str(uuid4())

    all_rounds: list[dict[str, Any]] = []

    first_round = await round_1(request_payload)
    all_rounds.append(first_round)
    save_round(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        prompt=display_prompt,
        round_data=first_round,
        rounds_expected=rounds,
        prompt_index=prompt_index,
        prompt_count=prompt_count,
        request_payload=request_payload,
        counterfactual=counterfactual,
    )

    for round_number in range(2, rounds + 1):
        nth_round = await round_n(request_payload, all_rounds, round_number)
        all_rounds.append(nth_round)
        save_round(
            batch_id=resolved_batch_id,
            prompt_id=resolved_prompt_id,
            prompt=display_prompt,
            round_data=nth_round,
            rounds_expected=rounds,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
            request_payload=request_payload,
            counterfactual=counterfactual,
        )

    synthesis_result = await synthesize(request_payload, all_rounds)
    stored = save_synthesis(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        prompt=display_prompt,
        synthesis=synthesis_result,
        rounds_expected=rounds,
        prompt_index=prompt_index,
        prompt_count=prompt_count,
        request_payload=request_payload,
        counterfactual=counterfactual,
    )

    evidence_index = build_evidence_index(stored)
    updated = update_prompt_result(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        updates={
            "evidence_index": evidence_index,
            "schema_version": 2,
        },
    )

    return updated


async def run_batch_deliberation(
    prompts: list[str] | None = None,
    items: list[dict[str, Any]] | None = None,
    rounds: int = ROUNDS,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Run deliberation sequentially for a batch of prompts/items."""
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

    for index, item in enumerate(normalized_items, start=1):
        item_rounds = int(item.get("rounds") or rounds)
        await run_deliberation(
            rounds=item_rounds,
            batch_id=resolved_batch_id,
            prompt_id=f"prompt-{index:04d}-{uuid4().hex[:8]}",
            prompt_index=index,
            prompt_count=total_prompts,
            deliberation_input=item,
        )

    stored_batch = load_result(resolved_batch_id) or {"batch": None, "results": []}
    return {
        "batch_id": resolved_batch_id,
        "rounds": rounds,
        "prompt_count": total_prompts,
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
) -> dict[str, Any]:
    """Run a full counterfactual rerun with selected evidence masked out."""
    source_result = load_prompt_result(source_batch_id, source_prompt_id)
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
    )
