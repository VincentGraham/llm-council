"""Inference client for local NIM containers with OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import COUNCIL_MODELS, MODEL_ENDPOINTS, MODEL_REQUEST_NAMES

logger = logging.getLogger(__name__)


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI-compatible content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


async def query_model(
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 300.0,
    endpoint: str | None = None,
) -> dict[str, Any] | None:
    """
    Query one local model using the OpenAI-compatible endpoint.

    Returns:
        Dict with `content` and optional `reasoning_details`, or `None` on failure.
    """
    base_url = endpoint or MODEL_ENDPOINTS.get(model)
    if not base_url:
        logger.error("Model endpoint not configured for %s", model)
        return None

    payload = {
        "model": MODEL_REQUEST_NAMES.get(model, model),
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Error querying model %s", model)
        return None

    choices = data.get("choices") or []
    if not choices:
        return {"content": "", "reasoning_details": None}

    message = choices[0].get("message") or {}
    return {
        "content": _message_content_to_text(message.get("content")),
        "reasoning_details": message.get("reasoning_details"),
    }


async def query_models_parallel(
    models: list[str],
    messages_by_model: dict[str, list[dict[str, str]]] | list[dict[str, str]],
    timeout: float = 300.0,
) -> dict[str, dict[str, Any] | None]:
    """Query multiple models in parallel."""
    tasks = []
    for model in models:
        if isinstance(messages_by_model, dict):
            model_messages = messages_by_model[model]
        else:
            model_messages = messages_by_model
        tasks.append(query_model(model=model, messages=model_messages, timeout=timeout))

    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}


async def health_check(model: str, timeout: float = 10.0) -> dict[str, Any]:
    """Run health check against one local model endpoint."""
    base_url = MODEL_ENDPOINTS.get(model)
    if not base_url:
        return {
            "model": model,
            "endpoint": None,
            "healthy": False,
            "error": "Model endpoint not configured",
        }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/models")
            response.raise_for_status()
            data = response.json()
            models = [
                item.get("id")
                for item in (data.get("data") or [])
                if isinstance(item, dict) and item.get("id")
            ]
        return {
            "model": model,
            "endpoint": base_url,
            "healthy": True,
            "available_models": models,
        }
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Health check failed for model %s: %s", model, exc)
        return {
            "model": model,
            "endpoint": base_url,
            "healthy": False,
            "error": str(exc),
        }


async def health_check_all(models: list[str] | None = None) -> list[dict[str, Any]]:
    """Run health checks for all council models."""
    selected = models or COUNCIL_MODELS
    checks = await asyncio.gather(*(health_check(model) for model in selected))
    return list(checks)
