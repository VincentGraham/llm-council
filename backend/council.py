"""N-round LLM council orchestration."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, ROUNDS
from .inference import query_model, query_models_parallel
from .storage import load_result, save_round, save_synthesis


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"Response {alphabet[index]}"
    return f"Response {index + 1}"


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
    """
    Calculate aggregate rankings from model reviews.

    Supports either one shared mapping, or per-review mappings attached as
    `result["label_to_model"]`.
    """
    model_positions: dict[str, list[int]] = defaultdict(list)

    for result in stage_results:
        ranking_text = (
            result.get("ranking")
            or result.get("review")
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


async def round_1(prompt: str) -> dict[str, Any]:
    """Round 1: all council models answer the prompt independently."""
    messages = [{"role": "user", "content": prompt}]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    round_responses = []
    for model in COUNCIL_MODELS:
        response = responses.get(model)
        if response is None:
            round_responses.append(
                {
                    "model": model,
                    "response": "",
                    "status": "error",
                    "error": "No response from model",
                }
            )
            continue
        round_responses.append(
            {
                "model": model,
                "response": response.get("content", ""),
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
    prompt: str,
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

    prompt_text = f"""You are participating in round {len(prior_rounds) + 1} of an LLM council.

Original user prompt:
{prompt}

You are shown only anonymized responses from OTHER models in earlier rounds.

{prior_text}

For ranking, focus on the MOST RECENT prior round responses:
{latest_text}

Your tasks:
1. Critique the other responses with concrete reasoning.
2. Provide your refined answer to the original user prompt.
3. End with a ranking section in EXACTLY this format:

FINAL RANKING:
1. Response X
2. Response Y
...

Only use the provided response labels in the final ranking."""

    return prompt_text, latest_label_to_model


async def round_n(
    prompt: str,
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
        prompt_text, label_map = _build_round_n_prompt(model, prompt, prior_rounds)
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
                    "parsed_ranking": [],
                    "label_to_model": label_maps[model],
                    "status": "error",
                    "error": "No response from model",
                }
            )
            continue

        review_text = response.get("content", "")
        round_responses.append(
            {
                "model": model,
                "response": review_text,
                "review": review_text,
                "parsed_ranking": parse_ranking_from_text(review_text),
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


async def synthesize(prompt: str, all_rounds: list[dict[str, Any]]) -> dict[str, Any]:
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
    chairman_prompt = f"""You are the chairman of an LLM council.

Original prompt:
{prompt}

Council deliberation rounds:
{rounds_text}

Synthesize a final answer for the user. Prioritize factual correctness,
clear reasoning, and actionable output when relevant."""

    response = await query_model(
        model=CHAIRMAN_MODEL,
        messages=[{"role": "user", "content": chairman_prompt}],
    )

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: unable to generate synthesis.",
            "status": "error",
            "created_at": _utcnow(),
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get("content", ""),
        "reasoning_details": response.get("reasoning_details"),
        "status": "ok",
        "created_at": _utcnow(),
    }


async def run_deliberation(
    prompt: str,
    rounds: int = ROUNDS,
    batch_id: str | None = None,
    prompt_id: str | None = None,
    prompt_index: int | None = None,
    prompt_count: int | None = None,
) -> dict[str, Any]:
    """Run full N-round deliberation and persist results."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    resolved_batch_id = batch_id or str(uuid4())
    resolved_prompt_id = prompt_id or str(uuid4())

    all_rounds: list[dict[str, Any]] = []

    first_round = await round_1(prompt)
    all_rounds.append(first_round)
    save_round(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        prompt=prompt,
        round_data=first_round,
        rounds_expected=rounds,
        prompt_index=prompt_index,
        prompt_count=prompt_count,
    )

    for round_number in range(2, rounds + 1):
        nth_round = await round_n(prompt, all_rounds, round_number)
        all_rounds.append(nth_round)
        save_round(
            batch_id=resolved_batch_id,
            prompt_id=resolved_prompt_id,
            prompt=prompt,
            round_data=nth_round,
            rounds_expected=rounds,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
        )

    synthesis_result = await synthesize(prompt, all_rounds)
    stored = save_synthesis(
        batch_id=resolved_batch_id,
        prompt_id=resolved_prompt_id,
        prompt=prompt,
        synthesis=synthesis_result,
        rounds_expected=rounds,
        prompt_index=prompt_index,
        prompt_count=prompt_count,
    )

    return stored


async def run_batch_deliberation(
    prompts: list[str],
    rounds: int = ROUNDS,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Run deliberation sequentially for a batch of prompts."""
    if not prompts:
        raise ValueError("prompts cannot be empty")

    resolved_batch_id = batch_id or str(uuid4())
    total_prompts = len(prompts)

    for index, prompt in enumerate(prompts, start=1):
        await run_deliberation(
            prompt=prompt,
            rounds=rounds,
            batch_id=resolved_batch_id,
            prompt_id=f"prompt-{index:04d}-{uuid4().hex[:8]}",
            prompt_index=index,
            prompt_count=total_prompts,
        )

    stored_batch = load_result(resolved_batch_id) or {"batch": None, "results": []}
    return {
        "batch_id": resolved_batch_id,
        "rounds": rounds,
        "prompt_count": total_prompts,
        "batch": stored_batch.get("batch"),
        "results": stored_batch.get("results", []),
    }
