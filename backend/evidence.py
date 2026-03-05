"""Structured evidence extraction, anchoring, selection, and masking."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from .config import EVIDENCE_MAX_ITEMS, EVIDENCE_MIN_ITEMS, EXTRACTOR_MODEL
from .inference import query_model


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_json_candidates(raw_text: str) -> list[str]:
    candidates = [match.group(1).strip() for match in JSON_BLOCK_RE.finditer(raw_text)]

    depth = 0
    in_string = False
    escape = False
    start = None
    for index, char in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw_text[start:index + 1].strip())
                    start = None

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def extract_json_block_deterministic(raw_text: str) -> dict[str, Any]:
    """Extract first valid JSON object from model output."""
    for candidate in _extract_json_candidates(raw_text):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in response.")


def _strip_json_block(raw_text: str) -> str:
    match = JSON_BLOCK_RE.search(raw_text)
    if not match:
        return raw_text.strip()
    cleaned = (raw_text[:match.start()] + raw_text[match.end():]).strip()
    return cleaned.strip()


def _normalize_prediction(structured_json: dict[str, Any]) -> dict[str, Any] | None:
    prediction_raw = structured_json.get("prediction")
    if not isinstance(prediction_raw, dict):
        prediction_raw = structured_json

    prediction = {
        "task_type": prediction_raw.get("task_type") or prediction_raw.get("type"),
        "value_text": prediction_raw.get("value_text") or prediction_raw.get("value"),
        "value_numeric": _to_float(
            prediction_raw.get("value_numeric")
            if prediction_raw.get("value_numeric") is not None
            else prediction_raw.get("numeric_value")
        ),
        "unit": prediction_raw.get("unit"),
        "probability": _to_float(prediction_raw.get("probability")),
        "label": prediction_raw.get("label"),
        "confidence": _to_float(prediction_raw.get("confidence")),
    }

    has_required = any(
        prediction.get(key) is not None and prediction.get(key) != ""
        for key in ("value_text", "value_numeric", "probability", "label")
    )
    if not has_required:
        return None
    return prediction


def _normalize_evidence(structured_json: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_raw = structured_json.get("evidence")
    if evidence_raw is None:
        evidence_raw = structured_json.get("evidence_items")
    if not isinstance(evidence_raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in evidence_raw[:EVIDENCE_MAX_ITEMS]:
        if isinstance(item, str):
            entry = {
                "quote": "",
                "rationale": item,
                "confidence": None,
                "source_tag": None,
            }
        elif isinstance(item, dict):
            entry = {
                "quote": str(item.get("quote") or item.get("span") or item.get("text") or "").strip(),
                "rationale": str(
                    item.get("rationale")
                    or item.get("reasoning")
                    or item.get("reason")
                    or ""
                ).strip(),
                "confidence": _to_float(item.get("confidence")),
                "source_tag": str(item.get("source_tag") or item.get("tag") or "").strip() or None,
            }
        else:
            continue
        normalized.append(entry)
    return normalized


def _find_fuzzy_span(quote: str, source_text: str) -> tuple[int, int] | None:
    quote_norm = _normalize_whitespace(quote).lower()
    if not quote_norm:
        return None

    best: tuple[float, int, int] | None = None
    for match in re.finditer(r"[^.!?\n]+(?:[.!?\n]|$)", source_text):
        candidate = match.group(0).strip()
        if not candidate:
            continue
        candidate_norm = _normalize_whitespace(candidate).lower()
        score = SequenceMatcher(None, quote_norm, candidate_norm).ratio()
        if best is None or score > best[0]:
            best = (score, match.start(), match.end())

    if best and best[0] >= 0.84:
        return best[1], best[2]
    return None


def anchor_evidence_quotes(
    evidence: list[dict[str, Any]],
    source_text: str,
    allow_fuzzy_quotes: bool,
    evidence_id_prefix: str,
) -> list[dict[str, Any]]:
    """Attach source spans to evidence quotes for downstream masking."""
    anchored: list[dict[str, Any]] = []
    source = source_text or ""

    for index, item in enumerate(evidence, start=1):
        quote = item.get("quote") or ""
        evidence_id = f"{evidence_id_prefix}-ev-{index:02d}"
        span_start: int | None = None
        span_end: int | None = None
        match_type = "unmatched"

        if quote and source:
            exact_start = source.find(quote)
            if exact_start >= 0:
                span_start = exact_start
                span_end = exact_start + len(quote)
                match_type = "exact"
            elif allow_fuzzy_quotes:
                fuzzy_span = _find_fuzzy_span(quote, source)
                if fuzzy_span is not None:
                    span_start, span_end = fuzzy_span
                    match_type = "fuzzy"

        anchored.append(
            {
                "evidence_id": evidence_id,
                "quote": quote,
                "rationale": item.get("rationale"),
                "confidence": item.get("confidence"),
                "source_tag": item.get("source_tag"),
                "span_start": span_start,
                "span_end": span_end,
                "match_type": match_type,
                "maskable": span_start is not None and span_end is not None and span_end > span_start,
            }
        )

    return anchored


async def repair_json_with_extractor(
    raw_text: str,
    schema_hint: str,
    extractor_model: str = EXTRACTOR_MODEL,
) -> dict[str, Any]:
    """Use extractor model to repair/normalize JSON payload."""
    prompt = f"""Extract one valid JSON object from the model output below.
Return JSON only, no markdown and no additional text.

Target schema:
{schema_hint}

Model output:
{raw_text}
"""

    response = await query_model(
        model=extractor_model,
        messages=[{"role": "user", "content": prompt}],
        timeout=300.0,
    )
    if response is None:
        raise ValueError("Extractor model did not return a response.")
    repaired_text = response.get("content", "")
    return extract_json_block_deterministic(repaired_text)


async def parse_hybrid_output(
    raw_text: str,
    source_text: str,
    evidence_id_prefix: str,
    allow_fuzzy_quotes: bool = False,
    extractor_model: str = EXTRACTOR_MODEL,
) -> dict[str, Any]:
    """Parse narrative + JSON hybrid model output into normalized fields."""
    parse_errors: list[str] = []
    parse_status = "failed"
    structured_json: dict[str, Any] | None = None

    schema_hint = """{
  "prediction": {
    "task_type": "duration|price|success|other",
    "value_text": "string or null",
    "value_numeric": "number or null",
    "unit": "string or null",
    "probability": "number between 0 and 1 or null",
    "label": "string or null",
    "confidence": "number between 0 and 1 or null"
  },
  "evidence": [
    {
      "quote": "exact source quote if possible",
      "rationale": "why this quote supports the prediction",
      "confidence": "number between 0 and 1",
      "source_tag": "optional short tag"
    }
  ]
}"""

    try:
        structured_json = extract_json_block_deterministic(raw_text)
        parse_status = "parsed"
    except Exception as exc:  # pylint: disable=broad-except
        parse_errors.append(f"deterministic_parse_failed: {exc}")
        try:
            structured_json = await repair_json_with_extractor(
                raw_text=raw_text,
                schema_hint=schema_hint,
                extractor_model=extractor_model,
            )
            parse_status = "repaired"
        except Exception as repair_exc:  # pylint: disable=broad-except
            parse_errors.append(f"repair_parse_failed: {repair_exc}")

    prediction = None
    evidence = []
    if structured_json is not None:
        prediction = _normalize_prediction(structured_json)
        evidence = _normalize_evidence(structured_json)
        if len(evidence) < EVIDENCE_MIN_ITEMS:
            parse_errors.append(
                f"evidence_count_below_minimum: {len(evidence)} < {EVIDENCE_MIN_ITEMS}"
            )
        evidence = anchor_evidence_quotes(
            evidence=evidence,
            source_text=source_text,
            allow_fuzzy_quotes=allow_fuzzy_quotes,
            evidence_id_prefix=evidence_id_prefix,
        )

    narrative_text = _strip_json_block(raw_text)
    if not narrative_text:
        narrative_text = raw_text.strip()

    return {
        "response": narrative_text,
        "prediction": prediction,
        "evidence": evidence,
        "structured_json": structured_json,
        "structured_parse_status": parse_status,
        "structured_parse_errors": parse_errors,
    }


def build_evidence_index(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all evidence from rounds and synthesis into one list."""
    items: list[dict[str, Any]] = []
    batch_id = result.get("batch_id")
    prompt_id = result.get("prompt_id")

    for round_data in result.get("rounds", []):
        round_number = round_data.get("round")
        for response in round_data.get("responses", []):
            model = response.get("model")
            for evidence in response.get("evidence", []):
                items.append(
                    {
                        "evidence_id": evidence.get("evidence_id"),
                        "batch_id": batch_id,
                        "prompt_id": prompt_id,
                        "model": model,
                        "round": round_number,
                        "is_synthesis": False,
                        "quote": evidence.get("quote"),
                        "rationale": evidence.get("rationale"),
                        "confidence": evidence.get("confidence"),
                        "source_tag": evidence.get("source_tag"),
                        "span_start": evidence.get("span_start"),
                        "span_end": evidence.get("span_end"),
                        "match_type": evidence.get("match_type"),
                        "maskable": evidence.get("maskable", False),
                    }
                )

    round_syntheses = result.get("round_syntheses")
    if isinstance(round_syntheses, list) and round_syntheses:
        synthesis_entries = [
            {
                "round": entry.get("round"),
                "synthesis": entry.get("synthesis") or {},
            }
            for entry in round_syntheses
            if isinstance(entry, dict)
        ]
    else:
        synthesis_entries = [
            {
                "round": result.get("rounds_expected"),
                "synthesis": result.get("synthesis") or {},
            }
        ]

    for entry in synthesis_entries:
        synthesis = entry.get("synthesis") or {}
        synthesis_round = entry.get("round")
        for evidence in synthesis.get("evidence", []):
            items.append(
                {
                    "evidence_id": evidence.get("evidence_id"),
                    "batch_id": batch_id,
                    "prompt_id": prompt_id,
                    "model": synthesis.get("model"),
                    "round": synthesis_round,
                    "is_synthesis": True,
                    "synthesis_round": synthesis_round,
                    "quote": evidence.get("quote"),
                    "rationale": evidence.get("rationale"),
                    "confidence": evidence.get("confidence"),
                    "source_tag": evidence.get("source_tag"),
                    "span_start": evidence.get("span_start"),
                    "span_end": evidence.get("span_end"),
                    "match_type": evidence.get("match_type"),
                    "maskable": evidence.get("maskable", False),
                }
            )

    items.sort(
        key=lambda item: (
            item.get("is_synthesis", False),
            item.get("round") if item.get("round") is not None else 999999,
            item.get("model") or "",
            item.get("evidence_id") or "",
        )
    )
    return items


def select_evidence_items(
    evidence_index: list[dict[str, Any]],
    evidence_ids: list[str] | None = None,
    selectors: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select evidence via explicit IDs plus selector filters (union)."""
    selected_ids = set(evidence_ids or [])
    selectors = selectors or {}

    use_selector_filters = (
        bool(selectors.get("models"))
        or bool(selectors.get("rounds"))
        or bool(selectors.get("source_tags"))
        or selectors.get("include_synthesis") is not None
    )

    if not use_selector_filters:
        selected = [item for item in evidence_index if item.get("evidence_id") in selected_ids]
        selected.sort(key=lambda item: item.get("evidence_id") or "")
        return selected

    models_filter = set(selectors.get("models") or [])
    rounds_filter = set(selectors.get("rounds") or [])
    source_tags_filter = set(selectors.get("source_tags") or [])
    include_synthesis = selectors.get("include_synthesis")

    for item in evidence_index:
        matches = True
        if models_filter and item.get("model") not in models_filter:
            matches = False
        if rounds_filter and item.get("round") not in rounds_filter:
            matches = False
        if source_tags_filter and item.get("source_tag") not in source_tags_filter:
            matches = False
        if include_synthesis is False and item.get("is_synthesis"):
            matches = False
        if matches:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                selected_ids.add(evidence_id)

    selected = [item for item in evidence_index if item.get("evidence_id") in selected_ids]
    selected.sort(key=lambda item: item.get("evidence_id") or "")
    return selected


def mask_source_text(
    source_text: str,
    selected_items: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], list[str]]:
    """Mask selected spans with placeholders and return mapping manifest."""
    maskable = [
        item for item in selected_items
        if item.get("maskable")
        and isinstance(item.get("span_start"), int)
        and isinstance(item.get("span_end"), int)
        and item["span_end"] > item["span_start"]
    ]
    if not maskable:
        return source_text, {}, []

    maskable.sort(key=lambda item: (item["span_start"], item["span_end"]))
    merged: list[dict[str, Any]] = []
    for item in maskable:
        start = item["span_start"]
        end = item["span_end"]
        evidence_id = item.get("evidence_id")
        if not merged or start > merged[-1]["span_end"]:
            merged.append({
                "span_start": start,
                "span_end": end,
                "evidence_ids": [evidence_id] if evidence_id else [],
            })
        else:
            merged[-1]["span_end"] = max(merged[-1]["span_end"], end)
            if evidence_id:
                merged[-1]["evidence_ids"].append(evidence_id)

    replacement_plan = []
    mask_manifest: dict[str, Any] = {}
    for index, interval in enumerate(merged, start=1):
        placeholder = f"[EVIDENCE_{index:03d}]"
        evidence_ids = sorted(set(interval["evidence_ids"]))
        replacement_plan.append((interval["span_start"], interval["span_end"], placeholder))
        mask_manifest[placeholder] = {
            "span_start": interval["span_start"],
            "span_end": interval["span_end"],
            "evidence_ids": evidence_ids,
            "original_text": source_text[interval["span_start"]:interval["span_end"]],
        }

    masked_text = source_text
    for start, end, placeholder in sorted(replacement_plan, key=lambda x: x[0], reverse=True):
        masked_text = masked_text[:start] + placeholder + masked_text[end:]

    masked_ids = sorted({eid for item in maskable for eid in [item.get("evidence_id")] if eid})
    return masked_text, mask_manifest, masked_ids
