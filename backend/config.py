"""Configuration loader for local NIM/vLLM council."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from .config_schema import load_council_config

logger = logging.getLogger(__name__)

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML_PATH = Path(os.getenv("MODELS_YAML") or (PROJECT_ROOT / "models.yaml"))
DATA_DIR = os.getenv("DATA_DIR", "/data/results")
EVIDENCE_MIN_ITEMS = 3
EVIDENCE_MAX_ITEMS = 5


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODEL_ENDPOINT_HOST = os.getenv("MODEL_ENDPOINT_HOST", "localhost")
MODEL_ENDPOINT_SCHEME = os.getenv("MODEL_ENDPOINT_SCHEME", "http")

_MODELS_CONFIG = None
_CONFIG_LOAD_ERROR: Exception | None = None
try:
    _MODELS_CONFIG = load_council_config(MODELS_YAML_PATH)
except Exception as exc:  # pylint: disable=broad-except
    _CONFIG_LOAD_ERROR = exc
    logger.warning(
        "Failed to load models config from %s. Runtime operations will fail until fixed.",
        MODELS_YAML_PATH,
        exc_info=exc,
    )

if _MODELS_CONFIG is not None:
    MODELS = [model.model_dump() for model in _MODELS_CONFIG.models]
    ROUNDS = _MODELS_CONFIG.rounds

    COUNCIL_MODELS = [model["name"] for model in MODELS]
    MODEL_CONFIG_BY_NAME = {model["name"]: model for model in MODELS}
else:
    MODELS = []
    ROUNDS = int(os.getenv("ROUNDS", "3"))
    COUNCIL_MODELS = []
    MODEL_CONFIG_BY_NAME = {}


def _endpoint_env_var_name(model_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).upper().strip("_")
    return f"MODEL_ENDPOINT_{normalized}"


MODEL_ENDPOINTS = {}
for model in MODELS:
    env_override = os.getenv(_endpoint_env_var_name(model["name"]))
    fallback = f"{MODEL_ENDPOINT_SCHEME}://{MODEL_ENDPOINT_HOST}:{model['port']}/v1"
    if env_override is not None:
        MODEL_ENDPOINTS[model["name"]] = env_override
    else:
        MODEL_ENDPOINTS[model["name"]] = model.get("endpoint") or fallback

MODEL_REQUEST_NAMES = {
    model["name"]: model.get("request_model", model["name"])
    for model in MODELS
}

if MODELS:
    _chairmen = [model["name"] for model in MODELS if model.get("chairman")]
    if _chairmen:
        CHAIRMAN_MODEL = _chairmen[0]
    else:
        CHAIRMAN_MODEL = COUNCIL_MODELS[0]
else:
    CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "")

if _MODELS_CONFIG is not None:
    EXTRACTOR_MODEL = _MODELS_CONFIG.extractor_model or CHAIRMAN_MODEL

    OBSERVER_CHAIRMAN_MODE = _MODELS_CONFIG.deliberation.observer_chairman
    SHARE_SYNTHESIS_WITH_MEMBERS = _MODELS_CONFIG.deliberation.share_synthesis_with_members
    EARLY_STOP_ENABLED_DEFAULT = _MODELS_CONFIG.deliberation.early_stopping
    EARLY_STOP_MIN_ROUNDS = _MODELS_CONFIG.deliberation.min_rounds_before_stop
    SYNTHESIS_SIMILARITY_THRESHOLD = _MODELS_CONFIG.deliberation.synthesis_similarity_threshold
    CONSENSUS_RATIO_THRESHOLD = _MODELS_CONFIG.deliberation.consensus_ratio_threshold

    ROUND1_INFERENCE_PARAMS = _MODELS_CONFIG.inference.round1.model_dump(exclude_none=True)
    ROUND_N_INFERENCE_PARAMS = _MODELS_CONFIG.inference.round_n.model_dump(exclude_none=True)
    SYNTHESIS_INFERENCE_PARAMS = _MODELS_CONFIG.inference.synthesis.model_dump(exclude_none=True)
    EXTRACTOR_INFERENCE_PARAMS = _MODELS_CONFIG.inference.extractor.model_dump(exclude_none=True)
else:
    EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL") or CHAIRMAN_MODEL
    OBSERVER_CHAIRMAN_MODE = _parse_bool_env("OBSERVER_CHAIRMAN_MODE", True)
    SHARE_SYNTHESIS_WITH_MEMBERS = _parse_bool_env("SHARE_SYNTHESIS_WITH_MEMBERS", False)
    EARLY_STOP_ENABLED_DEFAULT = _parse_bool_env("EARLY_STOP_ENABLED_DEFAULT", True)
    EARLY_STOP_MIN_ROUNDS = int(os.getenv("EARLY_STOP_MIN_ROUNDS", "2"))
    SYNTHESIS_SIMILARITY_THRESHOLD = float(os.getenv("SYNTHESIS_SIMILARITY_THRESHOLD", "0.985"))
    CONSENSUS_RATIO_THRESHOLD = float(os.getenv("CONSENSUS_RATIO_THRESHOLD", "0.8"))

    ROUND1_INFERENCE_PARAMS = {"temperature": 0.7, "max_tokens": 2200}
    ROUND_N_INFERENCE_PARAMS = {"temperature": 0.5, "max_tokens": 2200}
    SYNTHESIS_INFERENCE_PARAMS = {"temperature": 0.25, "max_tokens": 1800}
    EXTRACTOR_INFERENCE_PARAMS = {"temperature": 0.0, "max_tokens": 1200}


def ensure_runtime_config() -> None:
    """Raise a clear runtime error when config failed to load."""
    if _CONFIG_LOAD_ERROR is not None:
        raise RuntimeError(
            f"Failed to load models configuration from {MODELS_YAML_PATH}: {_CONFIG_LOAD_ERROR}"
        ) from _CONFIG_LOAD_ERROR
    if not COUNCIL_MODELS:
        raise RuntimeError("No council models are configured.")
