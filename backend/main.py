"""FastAPI backend for local multi-round LLM council."""

from __future__ import annotations

import asyncio
import copy
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, ROUNDS
from .council import (
    run_batch_deliberation,
    run_counterfactual_deliberation,
    run_deliberation,
)
from .evidence import build_evidence_index
from .inference import close_shared_client, health_check_all
from .storage import load_prompt_result, load_result


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_shared_client()


app = FastAPI(title="LLM Council API", lifespan=lifespan)
MAX_PROMPT_CHARS = max(1, int(os.getenv("MAX_PROMPT_CHARS", "20000")))
MAX_TRIAL_TEXT_CHARS = max(1, int(os.getenv("MAX_TRIAL_TEXT_CHARS", "500000")))
HEALTH_CACHE_TTL_SECONDS = max(0.0, float(os.getenv("HEALTH_CACHE_TTL_SECONDS", "5")))
_HEALTH_CACHE_LOCK = asyncio.Lock()
_HEALTH_CACHE: dict[tuple[str, ...], tuple[float, list[dict[str, Any]]]] = {}


def _validate_payload_sizes(payload: DeliberationInputRequest) -> None:
    if payload.prompt and len(payload.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"'prompt' exceeds max length of {MAX_PROMPT_CHARS} characters.",
        )
    if payload.trial_text and len(payload.trial_text) > MAX_TRIAL_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"'trial_text' exceeds max length of {MAX_TRIAL_TEXT_CHARS} characters.",
        )


async def _cached_health_check(models: list[str]) -> list[dict[str, Any]]:
    key = tuple(models)
    now = time.monotonic()
    async with _HEALTH_CACHE_LOCK:
        cached = _HEALTH_CACHE.get(key)
        if cached is not None and (now - cached[0]) <= HEALTH_CACHE_TTL_SECONDS:
            return copy.deepcopy(cached[1])

    statuses = await health_check_all(models)
    async with _HEALTH_CACHE_LOCK:
        _HEALTH_CACHE[key] = (time.monotonic(), statuses)
    return copy.deepcopy(statuses)


class DeliberationInputRequest(BaseModel):
    """Request payload for one deliberation input."""

    prompt: str | None = None
    trial_text: str | None = None
    prediction_target: str | None = None
    rounds: int | None = Field(default=None, ge=1)
    allow_fuzzy_quotes: bool = False
    early_stopping: bool | None = None
    min_rounds_before_stop: int | None = Field(default=None, ge=1)
    share_synthesis_with_members: bool | None = None
    inference: dict[str, dict[str, float | int | None]] | None = None
    metadata: dict[str, Any] | None = None


class DeliberateRequest(DeliberationInputRequest):
    """Request body for single-prompt deliberation."""


class BatchRequest(BaseModel):
    """Request body for batch deliberation."""

    prompts: list[str] | None = None
    items: list[DeliberationInputRequest] | None = None
    rounds: int | None = Field(default=None, ge=1)


class CounterfactualSelectors(BaseModel):
    """Evidence selector filters for counterfactual masking."""

    models: list[str] | None = None
    rounds: list[int] | None = None
    source_tags: list[str] | None = None
    include_synthesis: bool | None = None


class CounterfactualRequest(BaseModel):
    """Request body for counterfactual reruns."""

    source_batch_id: str
    source_prompt_id: str
    evidence_ids: list[str] | None = None
    selectors: CounterfactualSelectors | None = None
    rounds: int | None = Field(default=None, ge=1)
    allow_fuzzy_quotes: bool = False
    metadata: dict[str, Any] | None = None


@app.get("/")
async def root() -> dict[str, str]:
    """Basic service info endpoint."""
    return {"status": "ok", "service": "LLM Council API", "mode": "local-nim"}


@app.post("/api/deliberate")
async def deliberate(request: DeliberateRequest) -> dict[str, Any]:
    """Run N-round deliberation for one prompt and return full stored result."""
    if not (request.prompt and request.prompt.strip()) and not (request.trial_text and request.trial_text.strip()):
        raise HTTPException(status_code=400, detail="Provide at least one of 'prompt' or 'trial_text'.")
    _validate_payload_sizes(request)

    deliberation_input = request.model_dump(exclude_none=True)

    try:
        return await run_deliberation(
            rounds=request.rounds or ROUNDS,
            deliberation_input=deliberation_input,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batch")
async def deliberate_batch(request: BatchRequest) -> dict[str, Any]:
    """Run N-round deliberation sequentially for a batch of prompts/items."""
    has_prompts = bool(request.prompts)
    has_items = bool(request.items)
    if has_prompts == has_items:
        raise HTTPException(status_code=400, detail="Provide exactly one of 'prompts' or 'items'.")

    prompts = None
    items = None
    if request.prompts:
        prompts = [prompt.strip() for prompt in request.prompts if isinstance(prompt, str) and prompt.strip()]
        if not prompts:
            raise HTTPException(status_code=400, detail="prompts must include at least one non-empty item")
        for prompt in prompts:
            _validate_payload_sizes(DeliberationInputRequest(prompt=prompt))
    elif request.items:
        items = [item.model_dump(exclude_none=True) for item in request.items]
        for item in request.items:
            _validate_payload_sizes(item)
        valid_items = [item for item in items if item.get("prompt") or item.get("trial_text")]
        if not valid_items:
            raise HTTPException(
                status_code=400,
                detail="items must include at least one entry with 'prompt' or 'trial_text'.",
            )
        items = valid_items

    try:
        return await run_batch_deliberation(
            prompts=prompts,
            items=items,
            rounds=request.rounds or ROUNDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/counterfactual")
async def counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
    """Mask selected evidence and rerun the full council."""
    try:
        return await run_counterfactual_deliberation(
            source_batch_id=request.source_batch_id,
            source_prompt_id=request.source_prompt_id,
            evidence_ids=request.evidence_ids,
            selectors=request.selectors.model_dump(exclude_none=True) if request.selectors else None,
            rounds=request.rounds,
            allow_fuzzy_quotes=request.allow_fuzzy_quotes,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/batch/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, Any]:
    """Load all prompt results for a stored batch."""
    result = await asyncio.to_thread(load_result, batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return result


@app.get("/api/evidence/{batch_id}/{prompt_id}")
async def get_evidence(batch_id: str, prompt_id: str) -> dict[str, Any]:
    """Return flattened evidence index for one stored prompt result."""
    result = await asyncio.to_thread(load_prompt_result, batch_id, prompt_id)
    if result is None:
        raise HTTPException(status_code=404, detail="prompt result not found")

    evidence_index = result.get("evidence_index") or build_evidence_index(result)
    return {
        "batch_id": batch_id,
        "prompt_id": prompt_id,
        "evidence_items": evidence_index,
        "count": len(evidence_index),
    }


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    """Return configured models with live health checks."""
    statuses = await _cached_health_check(COUNCIL_MODELS)
    return {
        "models": statuses,
        "chairman": CHAIRMAN_MODEL,
        "count": len(statuses),
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Return overall system readiness based on model endpoint health."""
    statuses = await _cached_health_check(COUNCIL_MODELS)
    all_healthy = bool(statuses) and all(status.get("healthy") for status in statuses)
    return {
        "ready": all_healthy,
        "model_count": len(statuses),
        "healthy_models": sum(1 for status in statuses if status.get("healthy")),
        "models": statuses,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
