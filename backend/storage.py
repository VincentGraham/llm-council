"""JSON storage for multi-round batch deliberation results."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Iterator

from .config import DATA_DIR

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


CURRENT_SCHEMA_VERSION = 2


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dir() -> Path:
    """Ensure result root directory exists and return it."""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _batch_dir(batch_id: str, create: bool = True) -> Path:
    path = ensure_data_dir() / batch_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _batch_manifest_path(batch_id: str, create: bool = True) -> Path:
    return _batch_dir(batch_id, create=create) / "batch.json"


def _prompt_result_path(batch_id: str, prompt_id: str, create: bool = True) -> Path:
    return _batch_dir(batch_id, create=create) / f"{prompt_id}.json"


@contextmanager
def _batch_lock(batch_id: str) -> Iterator[None]:
    """Inter-process lock for all read-modify-write operations in a batch."""
    lock_path = _batch_dir(batch_id, create=True) / ".batch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write to avoid partial/corrupt files under concurrency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2)

    fd, temp_path_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _prompt_preview(prompt: str, max_len: int = 120) -> str:
    clean = " ".join(prompt.split())
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len - 3]}..."


def _normalize_request(request_payload: dict[str, Any] | None, prompt: str) -> dict[str, Any]:
    payload = dict(request_payload or {})
    payload.setdefault("prompt", prompt)
    payload.setdefault("trial_text", None)
    payload.setdefault("prediction_target", None)
    payload.setdefault("allow_fuzzy_quotes", False)
    payload.setdefault("early_stopping", None)
    payload.setdefault("min_rounds_before_stop", None)
    payload.setdefault("metadata", None)
    return payload


def _normalize_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt", "")
    payload.setdefault("schema_version", 1)
    payload.setdefault("request", _normalize_request(None, prompt))
    payload.setdefault("evidence_index", [])
    payload.setdefault("counterfactual", None)
    payload.setdefault("round_syntheses", [])
    payload.setdefault("rounds", [])
    payload.setdefault("synthesis", None)
    payload.setdefault("rounds_expected", 1)
    payload.setdefault("actual_rounds", len(payload.get("rounds", [])))
    payload.setdefault("stopped_early", False)
    payload.setdefault("early_stop_reason", None)
    payload.setdefault("deliberation_meta", None)
    payload.setdefault("created_at", _utcnow())
    payload.setdefault("updated_at", payload["created_at"])
    return payload


def _load_or_create_manifest(batch_id: str) -> dict[str, Any]:
    path = _batch_manifest_path(batch_id, create=True)
    manifest = _read_json(path)
    if manifest is not None:
        return manifest
    now = _utcnow()
    manifest = {
        "batch_id": batch_id,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "prompt_count": 0,
        "prompts": [],
    }
    _write_json(path, manifest)
    return manifest


def _upsert_manifest_prompt(
    manifest: dict[str, Any],
    prompt_id: str,
    prompt: str,
    prompt_index: int | None,
    has_synthesis: bool,
) -> None:
    now = _utcnow()
    existing = None
    for item in manifest["prompts"]:
        if item["prompt_id"] == prompt_id:
            existing = item
            break

    if existing is None:
        manifest["prompts"].append(
            {
                "prompt_id": prompt_id,
                "prompt_index": prompt_index,
                "prompt_preview": _prompt_preview(prompt),
                "has_synthesis": has_synthesis,
                "updated_at": now,
            }
        )
    else:
        if prompt_index is not None:
            existing["prompt_index"] = prompt_index
        existing["prompt_preview"] = _prompt_preview(prompt)
        existing["has_synthesis"] = has_synthesis
        existing["updated_at"] = now

    manifest["prompt_count"] = len(manifest["prompts"])
    manifest["updated_at"] = now
    completed = sum(1 for item in manifest["prompts"] if item.get("has_synthesis"))
    manifest["status"] = "completed" if completed == manifest["prompt_count"] else "running"


def _load_or_create_prompt_result(
    batch_id: str,
    prompt_id: str,
    prompt: str,
    rounds_expected: int,
    prompt_index: int | None,
    prompt_count: int | None,
    request_payload: dict[str, Any] | None,
    counterfactual: dict[str, Any] | None,
) -> dict[str, Any]:
    path = _prompt_result_path(batch_id, prompt_id, create=True)
    payload = _read_json(path)
    if payload is not None:
        payload = _normalize_result_payload(payload)
        if prompt_index is not None:
            payload["prompt_index"] = prompt_index
        if prompt_count is not None:
            payload["prompt_count"] = prompt_count
        payload["rounds_expected"] = rounds_expected
        payload["prompt"] = prompt
        if request_payload is not None:
            payload["request"] = _normalize_request(request_payload, prompt)
        if counterfactual is not None:
            payload["counterfactual"] = counterfactual
        return payload

    now = _utcnow()
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "batch_id": batch_id,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_index": prompt_index,
        "prompt_count": prompt_count,
        "created_at": now,
        "updated_at": now,
        "rounds_expected": rounds_expected,
        "request": _normalize_request(request_payload, prompt),
        "evidence_index": [],
        "counterfactual": counterfactual,
        "round_syntheses": [],
        "rounds": [],
        "synthesis": None,
        "actual_rounds": 0,
        "stopped_early": False,
        "early_stop_reason": None,
        "deliberation_meta": None,
    }


def save_round(
    batch_id: str,
    prompt_id: str,
    prompt: str,
    round_data: dict[str, Any],
    rounds_expected: int,
    prompt_index: int | None = None,
    prompt_count: int | None = None,
    request_payload: dict[str, Any] | None = None,
    counterfactual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one deliberation round for a prompt result."""
    round_number = round_data.get("round")
    if not isinstance(round_number, int):
        raise ValueError("round_data must include integer 'round'.")

    with _batch_lock(batch_id):
        result = _load_or_create_prompt_result(
            batch_id=batch_id,
            prompt_id=prompt_id,
            prompt=prompt,
            rounds_expected=rounds_expected,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
            request_payload=request_payload,
            counterfactual=counterfactual,
        )

        result["schema_version"] = CURRENT_SCHEMA_VERSION
        result["rounds"] = [
            existing for existing in result["rounds"] if existing.get("round") != round_number
        ]
        result["rounds"].append(round_data)
        result["rounds"].sort(key=lambda item: item.get("round", 0))
        result["updated_at"] = _utcnow()

        _write_json(_prompt_result_path(batch_id, prompt_id, create=True), result)

        manifest = _load_or_create_manifest(batch_id)
        _upsert_manifest_prompt(
            manifest=manifest,
            prompt_id=prompt_id,
            prompt=prompt,
            prompt_index=prompt_index,
            has_synthesis=result.get("synthesis") is not None,
        )
        _write_json(_batch_manifest_path(batch_id, create=True), manifest)
        return result


def save_synthesis(
    batch_id: str,
    prompt_id: str,
    prompt: str,
    synthesis: dict[str, Any],
    rounds_expected: int,
    prompt_index: int | None = None,
    prompt_count: int | None = None,
    request_payload: dict[str, Any] | None = None,
    counterfactual: dict[str, Any] | None = None,
    evidence_index: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist chairman synthesis for a prompt result."""
    with _batch_lock(batch_id):
        result = _load_or_create_prompt_result(
            batch_id=batch_id,
            prompt_id=prompt_id,
            prompt=prompt,
            rounds_expected=rounds_expected,
            prompt_index=prompt_index,
            prompt_count=prompt_count,
            request_payload=request_payload,
            counterfactual=counterfactual,
        )

        result["schema_version"] = CURRENT_SCHEMA_VERSION
        result["synthesis"] = synthesis
        if evidence_index is not None:
            result["evidence_index"] = evidence_index
        result["updated_at"] = _utcnow()
        _write_json(_prompt_result_path(batch_id, prompt_id, create=True), result)

        manifest = _load_or_create_manifest(batch_id)
        _upsert_manifest_prompt(
            manifest=manifest,
            prompt_id=prompt_id,
            prompt=prompt,
            prompt_index=prompt_index,
            has_synthesis=True,
        )
        _write_json(_batch_manifest_path(batch_id, create=True), manifest)
        return result


def update_prompt_result(
    batch_id: str,
    prompt_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Update arbitrary top-level fields for one prompt result."""
    with _batch_lock(batch_id):
        payload = load_prompt_result(batch_id, prompt_id)
        if payload is None:
            raise ValueError(f"Prompt result not found: {batch_id}/{prompt_id}")

        payload.update(updates)
        payload["schema_version"] = CURRENT_SCHEMA_VERSION
        payload["updated_at"] = _utcnow()
        _write_json(_prompt_result_path(batch_id, prompt_id, create=True), payload)
        return payload


def load_prompt_result(batch_id: str, prompt_id: str) -> dict[str, Any] | None:
    """Load a single prompt result with backward-compatible normalization."""
    payload = _read_json(_prompt_result_path(batch_id, prompt_id, create=False))
    if payload is None:
        return None
    return _normalize_result_payload(payload)


def load_result(batch_id: str, prompt_id: str | None = None) -> dict[str, Any] | None:
    """
    Load result for one prompt, or load all prompt results for a batch.

    Returns:
        If prompt_id is set: per-prompt result object or None.
        If prompt_id is unset: dict with `batch` + `results` or None.
    """
    if prompt_id:
        return load_prompt_result(batch_id, prompt_id)

    batch_path = _batch_dir(batch_id, create=False)
    if not batch_path.exists():
        return None

    manifest = _read_json(_batch_manifest_path(batch_id, create=False))
    if manifest is None:
        return None

    result_files = [
        path
        for path in batch_path.glob("*.json")
        if path.name != "batch.json"
    ]
    results = []
    for path in result_files:
        payload = _read_json(path)
        if payload is not None:
            results.append(_normalize_result_payload(payload))

    results.sort(
        key=lambda item: (
            item.get("prompt_index") if item.get("prompt_index") is not None else 999999,
            item.get("created_at", ""),
        )
    )
    return {"batch": manifest, "results": results}


def list_batches() -> list[dict[str, Any]]:
    """List all stored batches with metadata summary."""
    batches: list[dict[str, Any]] = []
    data_dir = ensure_data_dir()

    for path in data_dir.iterdir():
        if not path.is_dir():
            continue

        manifest = _read_json(path / "batch.json")
        if manifest is None:
            continue

        completed = sum(1 for prompt in manifest.get("prompts", []) if prompt.get("has_synthesis"))
        batches.append(
            {
                "batch_id": manifest.get("batch_id", path.name),
                "created_at": manifest.get("created_at"),
                "updated_at": manifest.get("updated_at"),
                "status": manifest.get("status", "running"),
                "prompt_count": manifest.get("prompt_count", 0),
                "completed_prompts": completed,
            }
        )

    batches.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return batches
