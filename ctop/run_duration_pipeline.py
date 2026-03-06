#!/usr/bin/env python3
"""Run CTOP duration prediction with default per-evidence counterfactuals."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Awaitable, Callable
from uuid import uuid4

import pandas as pd

from backend.config import ROUNDS
from backend.council import run_counterfactual_deliberation, run_deliberation

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("/data/trials/output/trials_intermediate.parquet")
DEFAULT_OUTPUT_ROOT = Path("/data/trials/output")
DEFAULT_EPSILON_MONTHS = 1.0
DEFAULT_EMA_ALPHA = 0.12
DEFAULT_SHORT_THRESHOLD_MONTHS = 12.0
DEFAULT_CHECKPOINT_EVERY_TRIALS = 5

SEMANTIC_TEXT_FIELDS = [
    ("Brief title", "brief_title"),
    ("Summary", "summary_text"),
    ("Detailed description", "description_text"),
    ("Primary outcomes", "primary_outcomes_text"),
    ("Secondary outcomes", "secondary_outcomes_text"),
    ("Phase", "phase"),
    ("Overall status", "overall_status"),
    ("Conditions", "conditions_text"),
    ("Interventions", "interventions_text"),
]

LONG_OUTPUT_COLUMNS = [
    "run_id",
    "nct_id",
    "trial_index",
    "run_type",
    "cf_index",
    "masked_evidence_id",
    "batch_id",
    "prompt_id",
    "parent_batch_id",
    "parent_prompt_id",
    "prediction_task_type",
    "prediction_value_numeric",
    "prediction_value_text",
    "prediction_unit",
    "prediction_probability",
    "prediction_label",
    "prediction_confidence",
    "structured_parse_status",
    "status",
    "error_type",
    "error_message",
    "response_text",
    "true_duration_days",
    "true_duration_months",
]

WIDE_OUTPUT_COLUMNS = [
    "run_id",
    "nct_id",
    "baseline_status",
    "baseline_structured_parse_status",
    "baseline_error_type",
    "baseline_error_message",
    "baseline_prediction_task_type",
    "baseline_prediction_value_numeric",
    "baseline_prediction_value_text",
    "baseline_prediction_unit",
    "baseline_prediction_probability",
    "baseline_prediction_label",
    "baseline_prediction_confidence",
    "cf_total",
    "cf_success",
    "cf_failed",
    "cf_numeric_mean",
    "cf_numeric_min",
    "cf_numeric_max",
    "cf_delta_mean",
    "cf_details_json",
    "true_duration_days",
    "true_duration_months",
]


@dataclass
class OnlineTrialMetrics:
    """Online metrics for baseline trial predictions."""

    epsilon_months: float = DEFAULT_EPSILON_MONTHS
    ema_alpha: float = DEFAULT_EMA_ALPHA
    short_threshold_months: float = DEFAULT_SHORT_THRESHOLD_MONTHS
    evaluated_trials: int = 0
    hit_count: int = 0
    ema_hit_rate: float | None = None
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    true_values: list[float] = field(default_factory=list)
    pred_values: list[float] = field(default_factory=list)

    def update(self, pred_months: float | None, true_months: float | None) -> None:
        """Update metrics from one baseline prediction."""
        if pred_months is None or true_months is None:
            return

        self.evaluated_trials += 1
        error = abs(pred_months - true_months)
        hit = 1.0 if error < self.epsilon_months else 0.0
        self.hit_count += int(hit)
        self.abs_error_sum += error
        self.sq_error_sum += error * error
        self.true_values.append(true_months)
        self.pred_values.append(pred_months)

        if self.ema_hit_rate is None:
            self.ema_hit_rate = hit
        else:
            self.ema_hit_rate = (
                self.ema_alpha * hit + (1.0 - self.ema_alpha) * self.ema_hit_rate
            )

        true_short = true_months <= self.short_threshold_months
        pred_short = pred_months <= self.short_threshold_months
        if true_short and pred_short:
            self.tp += 1
        elif (not true_short) and pred_short:
            self.fp += 1
        elif true_short and (not pred_short):
            self.fn += 1
        else:
            self.tn += 1

    @property
    def hit_rate(self) -> float | None:
        if self.evaluated_trials == 0:
            return None
        return self.hit_count / self.evaluated_trials

    @property
    def mae(self) -> float | None:
        if self.evaluated_trials == 0:
            return None
        return self.abs_error_sum / self.evaluated_trials

    @property
    def rmse(self) -> float | None:
        if self.evaluated_trials == 0:
            return None
        return math.sqrt(self.sq_error_sum / self.evaluated_trials)

    @property
    def f1_short(self) -> float | None:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return None
        return (2 * self.tp) / denom

    def summary(self) -> dict[str, Any]:
        """Return complete summary dictionary."""
        return {
            "epsilon_months": self.epsilon_months,
            "ema_alpha": self.ema_alpha,
            "short_threshold_months": self.short_threshold_months,
            "evaluated_trials": self.evaluated_trials,
            "hit_rate_epsilon": self.hit_rate,
            "ema_hit_rate_epsilon": self.ema_hit_rate,
            "mae": self.mae,
            "rmse": self.rmse,
            "f1_short_threshold": self.f1_short,
            "c_index": _concordance_index(self.true_values, self.pred_values),
        }


class ProgressReporter:
    """Two-line tqdm display: progress bar + latest baseline response line."""

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.enabled = enabled and tqdm is not None and total > 0
        self._bar = None
        self._latest = None
        if not self.enabled:
            return

        self._bar = tqdm(
            total=total,
            desc="Trials",
            dynamic_ncols=True,
            position=0,
            leave=True,
        )
        self._latest = tqdm(
            total=0,
            bar_format="{desc}",
            dynamic_ncols=True,
            position=1,
            leave=True,
        )
        self._latest.set_description_str("Latest: waiting...")
        self._latest.refresh()

    def extend_total(self, units: int) -> None:
        if not self.enabled or self._bar is None or units <= 0:
            return
        self._bar.total = (self._bar.total or 0) + units
        self._bar.refresh()

    def advance_round(self, latest_line: str, metrics: OnlineTrialMetrics, units: int = 1) -> None:
        if not self.enabled or self._bar is None or self._latest is None:
            return

        postfix: dict[str, str] = {}
        if metrics.evaluated_trials > 0:
            postfix["ema@1m"] = f"{metrics.ema_hit_rate:.3f}" if metrics.ema_hit_rate is not None else "n/a"
            postfix["hit@1m"] = f"{metrics.hit_rate:.3f}" if metrics.hit_rate is not None else "n/a"
            postfix["mae"] = f"{metrics.mae:.3f}" if metrics.mae is not None else "n/a"
            postfix["rmse"] = f"{metrics.rmse:.3f}" if metrics.rmse is not None else "n/a"
            postfix["f1@12m"] = f"{metrics.f1_short:.3f}" if metrics.f1_short is not None else "n/a"
        self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(max(0, units))

        trimmed = " ".join(latest_line.split())
        if len(trimmed) > 220:
            trimmed = f"{trimmed[:217]}..."
        self._latest.set_description_str(f"Latest: {trimmed}")
        self._latest.refresh()

    def close(self) -> None:
        if not self.enabled:
            return
        if self._latest is not None:
            self._latest.close()
        if self._bar is not None:
            self._bar.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    # Normalize array-likes (numpy/pandas) into python containers/scalars first.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray, dict)):
        try:
            normalized = value.tolist()
        except Exception:  # pylint: disable=broad-except
            normalized = value
        if normalized is not value:
            return _clean_text(normalized)

    # Some model outputs return arrays/lists for fields expected to be scalar text.
    # Collapse these safely instead of letting `pd.isna` return array-like masks.
    if isinstance(value, (list, tuple, set)):
        for item in value:
            cleaned = _clean_text(item)
            if cleaned:
                return cleaned
        return None
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=True).strip()
        return text or None

    try:
        missing = pd.isna(value)
    except Exception:  # pylint: disable=broad-except
        missing = False

    if isinstance(missing, bool):
        if missing:
            return None
    else:
        try:
            if bool(missing.all()):
                return None
        except Exception:  # pylint: disable=broad-except
            pass

    text = str(value).strip()
    return text or None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _concordance_index(true_values: list[float], pred_values: list[float]) -> float | None:
    """Pairwise concordance index (AUC-like ranking metric for regression)."""
    if len(true_values) < 2 or len(true_values) != len(pred_values):
        return None

    concordant = 0
    discordant = 0
    tied = 0

    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            true_diff = true_values[i] - true_values[j]
            if true_diff == 0:
                continue
            pred_diff = pred_values[i] - pred_values[j]
            if pred_diff == 0:
                tied += 1
            elif true_diff * pred_diff > 0:
                concordant += 1
            else:
                discordant += 1

    comparable = concordant + discordant + tied
    if comparable == 0:
        return None
    return (concordant + 0.5 * tied) / comparable


def _format_latest_line(baseline_row: dict[str, Any]) -> str:
    """Compact one-line baseline status for progress second line."""
    nct_id = _clean_text(baseline_row.get("nct_id")) or "unknown-nct"
    status = _clean_text(baseline_row.get("status")) or "unknown"
    if status != "ok":
        error_message = _clean_text(baseline_row.get("error_message")) or "no error message"
        return f"{nct_id} status={status} error={error_message}"

    pred = _to_float(baseline_row.get("prediction_value_numeric"))
    true = _to_float(baseline_row.get("true_duration_months"))
    if pred is None or true is None:
        pred_text = _clean_text(baseline_row.get("prediction_value_text")) or "n/a"
        return f"{nct_id} status=ok pred_text={pred_text}"

    abs_err = abs(pred - true)
    return f"{nct_id} status=ok pred={pred:.2f}m true={true:.2f}m abs_err={abs_err:.2f}m"


def _ensure_writable_dir(path: Path, strict: bool = True) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {path}")

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Could not create directory '{path}': {exc}"
        if strict:
            raise RuntimeError(message) from exc
        logger.warning(message)
        return

    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path, delete=True, encoding="utf-8") as handle:
            handle.write("ctop_duration_write_check")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Directory '{path}' is not writable: {exc}"
        if strict:
            raise PermissionError(message) from exc
        logger.warning(message)


def _read_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataframe not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(input_path)

    raise ValueError(
        "Unsupported input file extension. Use one of: .parquet, .csv, .pkl, .pickle"
    )


def _build_enrollment_text(row: dict[str, Any]) -> str | None:
    enrollment_count = _to_int(row.get("enrollment_count"))
    enrollment_type = _clean_text(row.get("enrollment_type"))

    parts: list[str] = []
    if enrollment_count is not None:
        parts.append(str(enrollment_count))
    if enrollment_type:
        parts.append(f"type={enrollment_type}")
    return ", ".join(parts) if parts else None


def _build_trial_text(row: dict[str, Any]) -> str:
    sections: list[str] = []

    for label, field in SEMANTIC_TEXT_FIELDS:
        value = _clean_text(row.get(field))
        if value:
            sections.append(f"{label}: {value}")

    enrollment_text = _build_enrollment_text(row)
    if enrollment_text:
        sections.append(f"Enrollment: {enrollment_text}")

    return "\n\n".join(sections).strip()


def _extract_prediction_fields(result: dict[str, Any]) -> dict[str, Any]:
    synthesis = result.get("synthesis")
    if not isinstance(synthesis, dict):
        synthesis = {}

    prediction = synthesis.get("prediction")
    if not isinstance(prediction, dict):
        prediction = {}

    status = _clean_text(synthesis.get("status")) or "ok"
    structured_parse_status = _clean_text(synthesis.get("structured_parse_status"))
    response_text = _clean_text(synthesis.get("response"))

    return {
        "prediction_task_type": _clean_text(prediction.get("task_type")),
        "prediction_value_numeric": _to_float(prediction.get("value_numeric")),
        "prediction_value_text": _clean_text(prediction.get("value_text")),
        "prediction_unit": _clean_text(prediction.get("unit")),
        "prediction_probability": _to_float(prediction.get("probability")),
        "prediction_label": _clean_text(prediction.get("label")),
        "prediction_confidence": _to_float(prediction.get("confidence")),
        "structured_parse_status": structured_parse_status,
        "status": status,
        "response_text": response_text,
    }


def _base_row(
    *,
    run_id: str,
    nct_id: str,
    trial_index: int,
    run_type: str,
    cf_index: int | None,
    masked_evidence_id: str | None,
    batch_id: str | None,
    prompt_id: str | None,
    parent_batch_id: str | None,
    parent_prompt_id: str | None,
    true_duration_days: int | None,
    true_duration_months: float | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "nct_id": nct_id,
        "trial_index": trial_index,
        "run_type": run_type,
        "cf_index": cf_index,
        "masked_evidence_id": masked_evidence_id,
        "batch_id": batch_id,
        "prompt_id": prompt_id,
        "parent_batch_id": parent_batch_id,
        "parent_prompt_id": parent_prompt_id,
        "prediction_task_type": None,
        "prediction_value_numeric": None,
        "prediction_value_text": None,
        "prediction_unit": None,
        "prediction_probability": None,
        "prediction_label": None,
        "prediction_confidence": None,
        "structured_parse_status": None,
        "status": "error",
        "error_type": None,
        "error_message": None,
        "response_text": None,
        "true_duration_days": true_duration_days,
        "true_duration_months": true_duration_months,
    }


def _error_row(
    *,
    run_id: str,
    nct_id: str,
    trial_index: int,
    run_type: str,
    cf_index: int | None,
    masked_evidence_id: str | None,
    batch_id: str | None,
    prompt_id: str | None,
    parent_batch_id: str | None,
    parent_prompt_id: str | None,
    error_type: str,
    error_message: str,
    true_duration_days: int | None,
    true_duration_months: float | None,
) -> dict[str, Any]:
    row = _base_row(
        run_id=run_id,
        nct_id=nct_id,
        trial_index=trial_index,
        run_type=run_type,
        cf_index=cf_index,
        masked_evidence_id=masked_evidence_id,
        batch_id=batch_id,
        prompt_id=prompt_id,
        parent_batch_id=parent_batch_id,
        parent_prompt_id=parent_prompt_id,
        true_duration_days=true_duration_days,
        true_duration_months=true_duration_months,
    )
    row["structured_parse_status"] = "failed"
    row["error_type"] = error_type
    row["error_message"] = error_message
    return row


def _success_row(
    *,
    run_id: str,
    nct_id: str,
    trial_index: int,
    run_type: str,
    cf_index: int | None,
    masked_evidence_id: str | None,
    result: dict[str, Any],
    parent_batch_id: str | None,
    parent_prompt_id: str | None,
    true_duration_days: int | None,
    true_duration_months: float | None,
) -> dict[str, Any]:
    row = _base_row(
        run_id=run_id,
        nct_id=nct_id,
        trial_index=trial_index,
        run_type=run_type,
        cf_index=cf_index,
        masked_evidence_id=masked_evidence_id,
        batch_id=_clean_text(result.get("batch_id")),
        prompt_id=_clean_text(result.get("prompt_id")),
        parent_batch_id=parent_batch_id,
        parent_prompt_id=parent_prompt_id,
        true_duration_days=true_duration_days,
        true_duration_months=true_duration_months,
    )
    row.update(_extract_prediction_fields(result))
    row["error_type"] = None
    row["error_message"] = None
    return row


async def _run_trial(
    *,
    run_id: str,
    trial_index: int,
    row: dict[str, Any],
    prediction_target: str,
    rounds: int,
    counterfactual_enabled: bool,
    allow_fuzzy_quotes: bool,
    round_progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    extend_progress_total_callback: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    nct_id = _clean_text(row.get("nct_id")) or f"trial-{trial_index:05d}"
    true_duration_days = _to_int(row.get("duration_days"))
    true_duration_months = _to_float(row.get("duration_months"))
    trial_text = _build_trial_text(row)

    if not trial_text:
        return [
            _error_row(
                run_id=run_id,
                nct_id=nct_id,
                trial_index=trial_index,
                run_type="baseline",
                cf_index=None,
                masked_evidence_id=None,
                batch_id=None,
                prompt_id=None,
                parent_batch_id=None,
                parent_prompt_id=None,
                error_type="EmptyTrialTextError",
                error_message="No semantic content available for trial_text prompt.",
                true_duration_days=true_duration_days,
                true_duration_months=true_duration_months,
            )
        ]

    metadata = {"nct_id": nct_id, "run_id": run_id, "trial_index": trial_index}
    deliberation_input = {
        "trial_text": trial_text,
        "prediction_target": prediction_target,
        "allow_fuzzy_quotes": allow_fuzzy_quotes,
        "metadata": metadata,
    }

    rows: list[dict[str, Any]] = []

    async def _baseline_round_progress(event: dict[str, Any]) -> None:
        if round_progress_callback is None:
            return
        payload = dict(event)
        payload["run_type"] = "baseline"
        payload["nct_id"] = nct_id
        payload["trial_index"] = trial_index
        await round_progress_callback(payload)

    try:
        baseline_result = await run_deliberation(
            rounds=rounds,
            deliberation_input=deliberation_input,
            round_progress_callback=_baseline_round_progress if round_progress_callback else None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        rows.append(
            _error_row(
                run_id=run_id,
                nct_id=nct_id,
                trial_index=trial_index,
                run_type="baseline",
                cf_index=None,
                masked_evidence_id=None,
                batch_id=None,
                prompt_id=None,
                parent_batch_id=None,
                parent_prompt_id=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                true_duration_days=true_duration_days,
                true_duration_months=true_duration_months,
            )
        )
        return rows

    baseline_batch_id = _clean_text(baseline_result.get("batch_id"))
    baseline_prompt_id = _clean_text(baseline_result.get("prompt_id"))
    rows.append(
        _success_row(
            run_id=run_id,
            nct_id=nct_id,
            trial_index=trial_index,
            run_type="baseline",
            cf_index=None,
            masked_evidence_id=None,
            result=baseline_result,
            parent_batch_id=None,
            parent_prompt_id=None,
            true_duration_days=true_duration_days,
            true_duration_months=true_duration_months,
        )
    )

    if not counterfactual_enabled:
        return rows

    synthesis = baseline_result.get("synthesis")
    evidence_items: list[dict[str, Any]] = []
    if isinstance(synthesis, dict):
        evidence_raw = synthesis.get("evidence")
        if isinstance(evidence_raw, list):
            evidence_items = [item for item in evidence_raw if isinstance(item, dict)]

    if not evidence_items:
        return rows

    if extend_progress_total_callback is not None:
        expected_cf_items = sum(1 for evidence in evidence_items if _clean_text(evidence.get("evidence_id")))
        extend_progress_total_callback(expected_cf_items * rounds)

    if not baseline_batch_id or not baseline_prompt_id:
        for cf_index, evidence in enumerate(evidence_items, start=1):
            rows.append(
                _error_row(
                    run_id=run_id,
                    nct_id=nct_id,
                    trial_index=trial_index,
                    run_type="counterfactual",
                    cf_index=cf_index,
                    masked_evidence_id=_clean_text(evidence.get("evidence_id")),
                    batch_id=None,
                    prompt_id=None,
                    parent_batch_id=baseline_batch_id,
                    parent_prompt_id=baseline_prompt_id,
                    error_type="MissingBaselineIdentifiers",
                    error_message="Baseline result missing batch_id or prompt_id.",
                    true_duration_days=true_duration_days,
                    true_duration_months=true_duration_months,
                )
            )
        return rows

    for cf_index, evidence in enumerate(evidence_items, start=1):
        evidence_id = _clean_text(evidence.get("evidence_id"))
        if not evidence_id:
            rows.append(
                _error_row(
                    run_id=run_id,
                    nct_id=nct_id,
                    trial_index=trial_index,
                    run_type="counterfactual",
                    cf_index=cf_index,
                    masked_evidence_id=None,
                    batch_id=None,
                    prompt_id=None,
                    parent_batch_id=baseline_batch_id,
                    parent_prompt_id=baseline_prompt_id,
                    error_type="MissingEvidenceId",
                    error_message="Final synthesis evidence item has no evidence_id.",
                    true_duration_days=true_duration_days,
                    true_duration_months=true_duration_months,
                )
            )
            continue

        cf_metadata = {
            "nct_id": nct_id,
            "run_id": run_id,
            "trial_index": trial_index,
            "masked_evidence_id": evidence_id,
        }

        async def _counterfactual_round_progress(event: dict[str, Any]) -> None:
            if round_progress_callback is None:
                return
            payload = dict(event)
            payload["run_type"] = "counterfactual"
            payload["nct_id"] = nct_id
            payload["trial_index"] = trial_index
            payload["cf_index"] = cf_index
            payload["masked_evidence_id"] = evidence_id
            await round_progress_callback(payload)

        try:
            cf_result = await run_counterfactual_deliberation(
                source_batch_id=baseline_batch_id,
                source_prompt_id=baseline_prompt_id,
                evidence_ids=[evidence_id],
                rounds=rounds,
                allow_fuzzy_quotes=allow_fuzzy_quotes,
                metadata=cf_metadata,
                round_progress_callback=_counterfactual_round_progress if round_progress_callback else None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            rows.append(
                _error_row(
                    run_id=run_id,
                    nct_id=nct_id,
                    trial_index=trial_index,
                    run_type="counterfactual",
                    cf_index=cf_index,
                    masked_evidence_id=evidence_id,
                    batch_id=None,
                    prompt_id=None,
                    parent_batch_id=baseline_batch_id,
                    parent_prompt_id=baseline_prompt_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    true_duration_days=true_duration_days,
                    true_duration_months=true_duration_months,
                )
            )
            continue

        rows.append(
            _success_row(
                run_id=run_id,
                nct_id=nct_id,
                trial_index=trial_index,
                run_type="counterfactual",
                cf_index=cf_index,
                masked_evidence_id=evidence_id,
                result=cf_result,
                parent_batch_id=baseline_batch_id,
                parent_prompt_id=baseline_prompt_id,
                true_duration_days=true_duration_days,
                true_duration_months=true_duration_months,
            )
        )
    return rows


def _build_wide_predictions_df(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=WIDE_OUTPUT_COLUMNS)

    ordered = long_df.sort_values(by=["trial_index", "cf_index"], na_position="last")
    wide_rows: list[dict[str, Any]] = []

    for nct_id, group in ordered.groupby("nct_id", dropna=False):
        baseline_group = group[group["run_type"] == "baseline"]
        baseline = baseline_group.iloc[0] if not baseline_group.empty else None
        cf_group = group[group["run_type"] == "counterfactual"].copy()

        baseline_numeric = None
        if baseline is not None:
            baseline_numeric = _to_float(baseline.get("prediction_value_numeric"))

        cf_numeric_values: list[float] = []
        cf_details: list[dict[str, Any]] = []
        for _, cf_row in cf_group.iterrows():
            cf_numeric = _to_float(cf_row.get("prediction_value_numeric"))
            if (_clean_text(cf_row.get("status")) == "ok") and (cf_numeric is not None):
                cf_numeric_values.append(cf_numeric)
            cf_details.append(
                {
                    "cf_index": _to_int(cf_row.get("cf_index")),
                    "masked_evidence_id": _clean_text(cf_row.get("masked_evidence_id")),
                    "status": _clean_text(cf_row.get("status")),
                    "prediction_value_numeric": cf_numeric,
                    "prediction_value_text": _clean_text(cf_row.get("prediction_value_text")),
                    "batch_id": _clean_text(cf_row.get("batch_id")),
                    "prompt_id": _clean_text(cf_row.get("prompt_id")),
                    "error_type": _clean_text(cf_row.get("error_type")),
                    "error_message": _clean_text(cf_row.get("error_message")),
                }
            )

        cf_total = len(cf_group)
        cf_success = int((cf_group["status"] == "ok").sum()) if cf_total else 0
        cf_failed = cf_total - cf_success

        cf_numeric_mean = sum(cf_numeric_values) / len(cf_numeric_values) if cf_numeric_values else None
        cf_numeric_min = min(cf_numeric_values) if cf_numeric_values else None
        cf_numeric_max = max(cf_numeric_values) if cf_numeric_values else None

        cf_delta_mean = None
        if baseline_numeric is not None and cf_numeric_values:
            deltas = [value - baseline_numeric for value in cf_numeric_values]
            cf_delta_mean = sum(deltas) / len(deltas)

        run_id = _clean_text(group.iloc[0].get("run_id"))
        true_duration_days = _to_int(group.iloc[0].get("true_duration_days"))
        true_duration_months = _to_float(group.iloc[0].get("true_duration_months"))

        wide_rows.append(
            {
                "run_id": run_id,
                "nct_id": _clean_text(nct_id),
                "baseline_status": _clean_text(baseline.get("status")) if baseline is not None else None,
                "baseline_structured_parse_status": (
                    _clean_text(baseline.get("structured_parse_status")) if baseline is not None else None
                ),
                "baseline_error_type": _clean_text(baseline.get("error_type")) if baseline is not None else None,
                "baseline_error_message": (
                    _clean_text(baseline.get("error_message")) if baseline is not None else None
                ),
                "baseline_prediction_task_type": (
                    _clean_text(baseline.get("prediction_task_type")) if baseline is not None else None
                ),
                "baseline_prediction_value_numeric": (
                    _to_float(baseline.get("prediction_value_numeric")) if baseline is not None else None
                ),
                "baseline_prediction_value_text": (
                    _clean_text(baseline.get("prediction_value_text")) if baseline is not None else None
                ),
                "baseline_prediction_unit": (
                    _clean_text(baseline.get("prediction_unit")) if baseline is not None else None
                ),
                "baseline_prediction_probability": (
                    _to_float(baseline.get("prediction_probability")) if baseline is not None else None
                ),
                "baseline_prediction_label": (
                    _clean_text(baseline.get("prediction_label")) if baseline is not None else None
                ),
                "baseline_prediction_confidence": (
                    _to_float(baseline.get("prediction_confidence")) if baseline is not None else None
                ),
                "cf_total": cf_total,
                "cf_success": cf_success,
                "cf_failed": cf_failed,
                "cf_numeric_mean": cf_numeric_mean,
                "cf_numeric_min": cf_numeric_min,
                "cf_numeric_max": cf_numeric_max,
                "cf_delta_mean": cf_delta_mean,
                "cf_details_json": json.dumps(cf_details, ensure_ascii=True),
                "true_duration_days": true_duration_days,
                "true_duration_months": true_duration_months,
            }
        )

    return pd.DataFrame(wide_rows, columns=WIDE_OUTPUT_COLUMNS)


def _write_dataframes(
    *,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> dict[str, Path]:
    paths = {
        "long_csv": output_dir / f"duration_predictions_long{suffix}.csv",
        "long_parquet": output_dir / f"duration_predictions_long{suffix}.parquet",
        "long_pkl": output_dir / f"duration_predictions_long{suffix}.pkl",
        "wide_csv": output_dir / f"duration_predictions_wide{suffix}.csv",
        "wide_parquet": output_dir / f"duration_predictions_wide{suffix}.parquet",
        "wide_pkl": output_dir / f"duration_predictions_wide{suffix}.pkl",
    }

    long_df.to_csv(paths["long_csv"], index=False)
    wide_df.to_csv(paths["wide_csv"], index=False)
    long_df.to_pickle(paths["long_pkl"])
    wide_df.to_pickle(paths["wide_pkl"])

    try:
        long_df.to_parquet(paths["long_parquet"], index=False)
        wide_df.to_parquet(paths["wide_parquet"], index=False)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "Failed to write parquet outputs. Install a parquet engine such as pyarrow."
        ) from exc

    return paths


def _write_checkpoint_snapshot(
    *,
    run_id: str,
    output_dir: Path,
    trial_index: int,
    total_trials: int,
    rows_snapshot: list[dict[str, Any]],
    metrics_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Persist rolling checkpoint artifacts for crash recovery."""
    long_df = pd.DataFrame(rows_snapshot, columns=LONG_OUTPUT_COLUMNS)
    wide_df = _build_wide_predictions_df(long_df)
    checkpoint_paths = _write_dataframes(
        long_df=long_df,
        wide_df=wide_df,
        output_dir=output_dir,
        suffix=".checkpoint",
    )
    payload = {
        "run_id": run_id,
        "checkpoint_at_trial": trial_index,
        "total_trials": total_trials,
        "long_rows": len(long_df),
        "wide_rows": len(wide_df),
        "evaluation_metrics": metrics_snapshot,
        "output_files": {key: str(path) for key, path in checkpoint_paths.items()},
        "updated_at": _utcnow(),
    }
    manifest_path = output_dir / "run_manifest.checkpoint.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_manifest(
    *,
    run_id: str,
    input_path: Path,
    output_dir: Path,
    started_at: str,
    completed_at: str,
    prediction_target: str,
    rounds: int,
    counterfactual_enabled: bool,
    allow_fuzzy_quotes: bool,
    max_trials: int | None,
    trial_count_input: int,
    trial_count_processed: int,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    evaluation_metrics: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    baseline_rows = long_df[long_df["run_type"] == "baseline"] if not long_df.empty else long_df
    cf_rows = long_df[long_df["run_type"] == "counterfactual"] if not long_df.empty else long_df

    baseline_failed = int((baseline_rows["status"] != "ok").sum()) if not baseline_rows.empty else 0
    baseline_success = len(baseline_rows) - baseline_failed
    cf_failed = int((cf_rows["status"] != "ok").sum()) if not cf_rows.empty else 0
    cf_success = len(cf_rows) - cf_failed

    return {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "prediction_target": prediction_target,
        "rounds": rounds,
        "counterfactual_enabled": counterfactual_enabled,
        "allow_fuzzy_quotes": allow_fuzzy_quotes,
        "max_trials": max_trials,
        "trial_count_input": trial_count_input,
        "trial_count_processed": trial_count_processed,
        "long_rows": len(long_df),
        "wide_rows": len(wide_df),
        "baseline_success": baseline_success,
        "baseline_failed": baseline_failed,
        "counterfactual_rows": len(cf_rows),
        "counterfactual_success": cf_success,
        "counterfactual_failed": cf_failed,
        "evaluation_metrics": evaluation_metrics,
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }


async def run_duration_pipeline(
    *,
    input_df: pd.DataFrame,
    run_id: str,
    prediction_target: str,
    rounds: int,
    counterfactual_enabled: bool,
    allow_fuzzy_quotes: bool,
    epsilon_months: float = DEFAULT_EPSILON_MONTHS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    short_threshold_months: float = DEFAULT_SHORT_THRESHOLD_MONTHS,
    show_progress: bool = True,
    checkpoint_every_trials: int = DEFAULT_CHECKPOINT_EVERY_TRIALS,
    checkpoint_callback: Callable[[int, list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = OnlineTrialMetrics(
        epsilon_months=epsilon_months,
        ema_alpha=ema_alpha,
        short_threshold_months=short_threshold_months,
    )
    progress = ProgressReporter(total=len(input_df) * max(1, rounds), enabled=show_progress)

    try:
        for trial_index, (_, series) in enumerate(input_df.iterrows(), start=1):
            trial_dict = series.to_dict()
            nct_id = _clean_text(trial_dict.get("nct_id")) or f"trial-{trial_index:05d}"
            trial_progress_units = 0

            async def on_round_progress(event: dict[str, Any]) -> None:
                nonlocal trial_progress_units
                run_type = _clean_text(event.get("run_type")) or "baseline"
                round_number = _to_int(event.get("round")) or 0
                rounds_expected = _to_int(event.get("rounds_expected")) or rounds
                actual_rounds_so_far = _to_int(event.get("actual_rounds_so_far")) or round_number
                if run_type == "counterfactual":
                    cf_index = _to_int(event.get("cf_index"))
                    label = (
                        f"{nct_id} cf#{cf_index} round {round_number}/{rounds_expected}"
                        if cf_index is not None
                        else f"{nct_id} counterfactual round {round_number}/{rounds_expected}"
                    )
                else:
                    label = f"{nct_id} baseline round {round_number}/{rounds_expected}"
                progress.advance_round(latest_line=label, metrics=metrics, units=1)
                trial_progress_units += 1
                if bool(event.get("stopped_early")) and rounds_expected > actual_rounds_so_far:
                    skipped_units = rounds_expected - actual_rounds_so_far
                    progress.advance_round(
                        latest_line=f"{label} (early stop)",
                        metrics=metrics,
                        units=skipped_units,
                    )
                    trial_progress_units += skipped_units

            trial_rows = await _run_trial(
                run_id=run_id,
                trial_index=trial_index,
                row=trial_dict,
                prediction_target=prediction_target,
                rounds=rounds,
                counterfactual_enabled=counterfactual_enabled,
                allow_fuzzy_quotes=allow_fuzzy_quotes,
                round_progress_callback=on_round_progress if show_progress else None,
                extend_progress_total_callback=progress.extend_total if show_progress else None,
            )
            rows.extend(trial_rows)

            expected_counterfactual_runs = sum(
                1
                for row_item in trial_rows
                if row_item.get("run_type") == "counterfactual" and _clean_text(row_item.get("masked_evidence_id"))
            )
            planned_units = rounds + (expected_counterfactual_runs * rounds)
            missing_units = planned_units - trial_progress_units
            if missing_units > 0 and show_progress:
                progress.advance_round(
                    latest_line=f"{nct_id} progress reconcile",
                    metrics=metrics,
                    units=missing_units,
                )
                trial_progress_units += missing_units

            baseline_row = next(
                (row for row in trial_rows if row.get("run_type") == "baseline"),
                None,
            )
            if baseline_row is None:
                baseline_row = {
                    "nct_id": nct_id,
                    "status": "error",
                    "error_message": "Baseline row missing",
                }

            metrics.update(
                pred_months=_to_float(baseline_row.get("prediction_value_numeric")),
                true_months=_to_float(baseline_row.get("true_duration_months")),
            )
            progress.advance_round(
                latest_line=_format_latest_line(baseline_row),
                metrics=metrics,
                units=0,
            )

            if (
                checkpoint_callback is not None
                and checkpoint_every_trials > 0
                and trial_index % checkpoint_every_trials == 0
            ):
                checkpoint_callback(trial_index, list(rows), metrics.summary())
    finally:
        progress.close()

    long_df = pd.DataFrame(rows, columns=LONG_OUTPUT_COLUMNS)
    wide_df = _build_wide_predictions_df(long_df)
    return long_df, wide_df, metrics.summary()


async def execute_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utcnow()
    input_path = Path(args.input_path)
    output_root = Path(args.output_root)
    run_id = args.run_id or uuid4().hex[:8]
    rounds = int(args.rounds) if args.rounds is not None else ROUNDS
    checkpoint_every_trials = int(args.checkpoint_every_trials)
    strict = bool(args.strict)

    if args.max_trials is not None and int(args.max_trials) <= 0:
        raise ValueError("--max-trials must be >= 1 when provided.")
    if float(args.epsilon_months) <= 0:
        raise ValueError("--epsilon-months must be > 0.")
    if not 0 < float(args.ema_alpha) <= 1:
        raise ValueError("--ema-alpha must be in the interval (0, 1].")
    if float(args.short_threshold_months) <= 0:
        raise ValueError("--short-threshold-months must be > 0.")
    if checkpoint_every_trials < 0:
        raise ValueError("--checkpoint-every-trials must be >= 0.")

    _ensure_writable_dir(output_root, strict=strict)
    run_output_dir = output_root / run_id
    _ensure_writable_dir(run_output_dir, strict=strict)

    input_df = _read_dataframe(input_path)
    trial_count_input = len(input_df)
    if args.max_trials is not None:
        input_df = input_df.head(int(args.max_trials))

    total_trials = len(input_df)

    def checkpoint_callback(
        trial_index: int,
        rows_snapshot: list[dict[str, Any]],
        metrics_snapshot: dict[str, Any],
    ) -> None:
        if checkpoint_every_trials <= 0:
            return
        try:
            _write_checkpoint_snapshot(
                run_id=run_id,
                output_dir=run_output_dir,
                trial_index=trial_index,
                total_trials=total_trials,
                rows_snapshot=rows_snapshot,
                metrics_snapshot=metrics_snapshot,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Checkpoint write failed at trial %s: %s", trial_index, exc)

    long_df, wide_df, evaluation_metrics = await run_duration_pipeline(
        input_df=input_df,
        run_id=run_id,
        prediction_target=str(args.prediction_target),
        rounds=rounds,
        counterfactual_enabled=bool(args.counterfactual),
        allow_fuzzy_quotes=bool(args.allow_fuzzy_quotes),
        epsilon_months=float(args.epsilon_months),
        ema_alpha=float(args.ema_alpha),
        short_threshold_months=float(args.short_threshold_months),
        show_progress=bool(args.progress),
        checkpoint_every_trials=checkpoint_every_trials,
        checkpoint_callback=checkpoint_callback if checkpoint_every_trials > 0 else None,
    )

    output_paths = _write_dataframes(
        long_df=long_df,
        wide_df=wide_df,
        output_dir=run_output_dir,
    )

    manifest = _build_manifest(
        run_id=run_id,
        input_path=input_path,
        output_dir=run_output_dir,
        started_at=started_at,
        completed_at=_utcnow(),
        prediction_target=str(args.prediction_target),
        rounds=rounds,
        counterfactual_enabled=bool(args.counterfactual),
        allow_fuzzy_quotes=bool(args.allow_fuzzy_quotes),
        max_trials=int(args.max_trials) if args.max_trials is not None else None,
        trial_count_input=trial_count_input,
        trial_count_processed=len(input_df),
        long_df=long_df,
        wide_df=wide_df,
        evaluation_metrics=evaluation_metrics,
        output_paths=output_paths,
    )
    manifest_path = run_output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Run ID: %s", run_id)
    logger.info("Trials processed: %s", len(input_df))
    logger.info("Long rows: %s", len(long_df))
    logger.info("Wide rows: %s", len(wide_df))
    logger.info("Manifest: %s", manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input dataframe path (.parquet/.csv/.pkl). Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier. Default: random 8-char UUID.",
    )
    parser.add_argument(
        "--prediction-target",
        type=str,
        default="duration",
        help="Prediction target passed to the council. Default: duration.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Override number of deliberation rounds. Default: config value ({ROUNDS}).",
    )
    parser.add_argument(
        "--counterfactual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-evidence counterfactual runs. Default: enabled.",
    )
    parser.add_argument(
        "--allow-fuzzy-quotes",
        action="store_true",
        default=False,
        help="Allow fuzzy evidence quote anchoring during parsing.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on number of trials to run.",
    )
    parser.add_argument(
        "--epsilon-months",
        type=float,
        default=DEFAULT_EPSILON_MONTHS,
        help=f"Epsilon threshold for hit-accuracy metric in months (default: {DEFAULT_EPSILON_MONTHS}).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help=f"EMA smoothing factor for running hit-accuracy (default: {DEFAULT_EMA_ALPHA}).",
    )
    parser.add_argument(
        "--short-threshold-months",
        type=float,
        default=DEFAULT_SHORT_THRESHOLD_MONTHS,
        help=(
            "Threshold in months used for short-vs-long F1 analog "
            f"(default: {DEFAULT_SHORT_THRESHOLD_MONTHS})."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show two-line tqdm progress display (default: true).",
    )
    parser.add_argument(
        "--checkpoint-every-trials",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_TRIALS,
        help=(
            "Write rolling checkpoint artifacts every N trials (default: "
            f"{DEFAULT_CHECKPOINT_EVERY_TRIALS}, use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if output directories are unavailable/not writable. Default: true.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        default="INFO",
        help="Logging verbosity for pipeline output. Use NONE to disable logging entirely.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_level == "NONE":
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(levelname)s %(message)s",
        )
    asyncio.run(execute_pipeline(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
