"""FastAPI backend for local multi-round LLM council."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, ROUNDS
from .council import run_batch_deliberation, run_deliberation
from .inference import health_check_all
from .storage import load_result


app = FastAPI(title="LLM Council API")


class DeliberateRequest(BaseModel):
    """Request body for single-prompt deliberation."""

    prompt: str = Field(..., min_length=1)
    rounds: int | None = Field(default=None, ge=1)


class BatchRequest(BaseModel):
    """Request body for batch deliberation."""

    prompts: list[str] = Field(..., min_length=1)
    rounds: int | None = Field(default=None, ge=1)


@app.get("/")
async def root() -> dict[str, str]:
    """Basic service info endpoint."""
    return {"status": "ok", "service": "LLM Council API", "mode": "local-nim"}


@app.post("/api/deliberate")
async def deliberate(request: DeliberateRequest) -> dict[str, Any]:
    """Run N-round deliberation for one prompt and return full stored result."""
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    try:
        return await run_deliberation(prompt=prompt, rounds=request.rounds or ROUNDS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batch")
async def deliberate_batch(request: BatchRequest) -> dict[str, Any]:
    """Run N-round deliberation sequentially for a list of prompts."""
    prompts = [prompt.strip() for prompt in request.prompts if prompt.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="prompts must include at least one non-empty item")

    try:
        return await run_batch_deliberation(prompts=prompts, rounds=request.rounds or ROUNDS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/batch/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, Any]:
    """Load all prompt results for a stored batch."""
    result = load_result(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return result


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    """Return configured models with live health checks."""
    statuses = await health_check_all(COUNCIL_MODELS)
    return {
        "models": statuses,
        "chairman": CHAIRMAN_MODEL,
        "count": len(statuses),
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Return overall system readiness based on model endpoint health."""
    statuses = await health_check_all(COUNCIL_MODELS)
    all_healthy = all(status.get("healthy") for status in statuses)
    return {
        "ready": all_healthy,
        "model_count": len(statuses),
        "healthy_models": sum(1 for status in statuses if status.get("healthy")),
        "models": statuses,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
