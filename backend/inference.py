"""Inference client for local NIM containers with OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from .config import COUNCIL_MODELS, MODEL_ENDPOINTS, MODEL_REQUEST_NAMES

logger = logging.getLogger(__name__)
_CLIENT_LOCK = asyncio.Lock()
_SHARED_CLIENT: httpx.AsyncClient | None = None
_SHARED_CLIENT_CREATED_AT = 0.0
_SHARED_CLIENT_TTL_SECONDS = float(os.getenv("SHARED_HTTP_CLIENT_TTL_SECONDS", "600"))
_QUERY_RETRIES = max(0, int(os.getenv("MODEL_QUERY_RETRIES", "2")))
_QUERY_RETRY_BACKOFF_SECONDS = float(os.getenv("MODEL_QUERY_RETRY_BACKOFF_SECONDS", "0.5"))
_RESERVED_PAYLOAD_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "top_p",
    "response_format",
}


def _normalize_model_id(model_id: str) -> str:
    return "".join(char for char in model_id.lower() if char.isalnum())


def _match_requested_model(
    requested_model: str,
    available_models: list[str],
) -> tuple[bool, str | None, str]:
    if requested_model in available_models:
        return True, requested_model, "exact"

    requested_norm = _normalize_model_id(requested_model)
    for available in available_models:
        if _normalize_model_id(available) == requested_norm:
            return True, available, "normalized"

    return False, None, "missing"


def _build_shared_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0),
    )


async def _get_shared_client() -> httpx.AsyncClient:
    global _SHARED_CLIENT, _SHARED_CLIENT_CREATED_AT
    now = time.monotonic()
    if _SHARED_CLIENT is not None:
        age_seconds = now - _SHARED_CLIENT_CREATED_AT
        if age_seconds <= _SHARED_CLIENT_TTL_SECONDS:
            return _SHARED_CLIENT
        async with _CLIENT_LOCK:
            if _SHARED_CLIENT is not None and (time.monotonic() - _SHARED_CLIENT_CREATED_AT) > _SHARED_CLIENT_TTL_SECONDS:
                await _SHARED_CLIENT.aclose()
                _SHARED_CLIENT = None

    async with _CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            _SHARED_CLIENT = _build_shared_client()
            _SHARED_CLIENT_CREATED_AT = time.monotonic()
    return _SHARED_CLIENT


async def close_shared_client() -> None:
    """Close the shared async HTTP client."""
    global _SHARED_CLIENT, _SHARED_CLIENT_CREATED_AT
    async with _CLIENT_LOCK:
        if _SHARED_CLIENT is not None:
            await _SHARED_CLIENT.aclose()
            _SHARED_CLIENT = None
            _SHARED_CLIENT_CREATED_AT = 0.0


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _merge_extra_body(payload: dict[str, Any], extra_body: dict[str, Any] | None) -> dict[str, Any]:
    if not extra_body:
        return payload
    for key, value in extra_body.items():
        if key in _RESERVED_PAYLOAD_KEYS:
            logger.warning("Ignoring extra_body override for reserved payload key: %s", key)
            continue
        payload[key] = value
    return payload


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
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Query one local model using the OpenAI-compatible endpoint.

    Returns:
        Dict with `content`, optional `reasoning_details`, and optional `usage`,
        or `None` on failure.
    """
    base_url = endpoint or MODEL_ENDPOINTS.get(model)
    if not base_url:
        logger.error("Model endpoint not configured for %s", model)
        return None

    payload = {
        "model": MODEL_REQUEST_NAMES.get(model, model),
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    payload = _merge_extra_body(payload, extra_body)

    for attempt in range(_QUERY_RETRIES + 1):
        try:
            client = await _get_shared_client()
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            break
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            retryable = _is_retryable_http_status(status_code)
            if retryable and attempt < _QUERY_RETRIES:
                backoff = _QUERY_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "Retrying model query after HTTP %s for %s (attempt %s/%s, backoff=%.2fs).",
                    status_code,
                    model,
                    attempt + 1,
                    _QUERY_RETRIES + 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            logger.warning("Model query failed for %s with HTTP %s", model, status_code)
            return None
        except (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError) as exc:
            if attempt < _QUERY_RETRIES:
                backoff = _QUERY_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "Retrying model query for %s after transport error %s (attempt %s/%s, backoff=%.2fs).",
                    model,
                    type(exc).__name__,
                    attempt + 1,
                    _QUERY_RETRIES + 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            logger.warning("Transport error querying model %s: %s", model, exc)
            return None
        except ValueError:
            logger.exception("Invalid JSON response while querying model %s", model)
            return None
        except Exception:  # pylint: disable=broad-except
            logger.exception("Unexpected error querying model %s", model)
            return None

    choices = data.get("choices") or []
    if not choices:
        return {"content": "", "reasoning_details": None, "usage": data.get("usage")}

    message = choices[0].get("message") or {}
    return {
        "content": _message_content_to_text(message.get("content")),
        "reasoning_details": message.get("reasoning_details"),
        "usage": data.get("usage"),
    }


async def query_models_parallel(
    models: list[str],
    messages_by_model: dict[str, list[dict[str, str]]] | list[dict[str, str]],
    timeout: float = 300.0,
    generation_params: dict[str, Any] | None = None,
    generation_params_by_model: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Query multiple models in parallel."""
    tasks = []
    for model in models:
        if isinstance(messages_by_model, dict):
            model_messages = messages_by_model[model]
        else:
            model_messages = messages_by_model

        model_params: dict[str, Any] = {}
        if generation_params:
            model_params.update(generation_params)
        if generation_params_by_model and generation_params_by_model.get(model):
            model_params.update(generation_params_by_model[model])

        tasks.append(
            query_model(
                model=model,
                messages=model_messages,
                timeout=timeout,
                **model_params,
            )
        )

    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}


async def health_check(model: str, timeout: float = 10.0) -> dict[str, Any]:
    """Run health check against one local model endpoint."""
    base_url = MODEL_ENDPOINTS.get(model)
    if not base_url:
        return {
            "model": model,
            "endpoint": None,
            "endpoint_healthy": False,
            "healthy": False,
            "error": "Model endpoint not configured",
        }

    try:
        client = await _get_shared_client()
        response = await client.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        available_models = [
            item.get("id")
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        request_model = MODEL_REQUEST_NAMES.get(model, model)
        request_model_available, matched_model_id, match_type = _match_requested_model(
            request_model,
            available_models,
        )
        healthy = bool(request_model_available)
        if not request_model_available:
            logger.warning(
                "Health check model mismatch for %s: request_model=%s, available=%s",
                model,
                request_model,
                available_models,
            )
        return {
            "model": model,
            "endpoint": base_url,
            "endpoint_healthy": True,
            "healthy": healthy,
            "request_model": request_model,
            "request_model_available": request_model_available,
            "request_model_match_type": match_type,
            "matched_model_id": matched_model_id,
            "available_models": available_models,
            "error": None if healthy else "Configured request_model was not found on endpoint",
        }
    except asyncio.CancelledError:
        raise
    except httpx.HTTPError as exc:
        logger.warning("Health check failed for model %s: %s", model, exc)
        return {
            "model": model,
            "endpoint": base_url,
            "endpoint_healthy": False,
            "healthy": False,
            "error": str(exc),
        }
    except ValueError as exc:
        logger.warning("Health check failed for model %s: %s", model, exc)
        return {
            "model": model,
            "endpoint": base_url,
            "endpoint_healthy": False,
            "healthy": False,
            "error": str(exc),
        }


async def health_check_all(models: list[str] | None = None) -> list[dict[str, Any]]:
    """Run health checks for all council models."""
    selected = models or COUNCIL_MODELS
    checks = await asyncio.gather(*(health_check(model) for model in selected))
    return list(checks)
