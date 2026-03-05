"""Configuration loader for local NIM/vLLM council."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from .config_schema import load_council_config


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML_PATH = Path(os.getenv("MODELS_YAML", PROJECT_ROOT / "models.yaml"))
DATA_DIR = os.getenv("DATA_DIR", "/data/results")
EVIDENCE_MIN_ITEMS = 3
EVIDENCE_MAX_ITEMS = 5


_MODELS_CONFIG = load_council_config(MODELS_YAML_PATH)

MODELS = [model.model_dump() for model in _MODELS_CONFIG.models]
ROUNDS = _MODELS_CONFIG.rounds

COUNCIL_MODELS = [model["name"] for model in MODELS]
MODEL_CONFIG_BY_NAME = {model["name"]: model for model in MODELS}
MODEL_ENDPOINT_HOST = os.getenv("MODEL_ENDPOINT_HOST", "localhost")
MODEL_ENDPOINT_SCHEME = os.getenv("MODEL_ENDPOINT_SCHEME", "http")


def _endpoint_env_var_name(model_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).upper().strip("_")
    return f"MODEL_ENDPOINT_{normalized}"


MODEL_ENDPOINTS = {}
for model in MODELS:
    env_override = os.getenv(_endpoint_env_var_name(model["name"]))
    fallback = f"{MODEL_ENDPOINT_SCHEME}://{MODEL_ENDPOINT_HOST}:{model['port']}/v1"
    MODEL_ENDPOINTS[model["name"]] = env_override or model.get("endpoint") or fallback

MODEL_REQUEST_NAMES = {
    model["name"]: model.get("request_model", model["name"])
    for model in MODELS
}

_chairmen = [model["name"] for model in MODELS if model.get("chairman")]
if _chairmen:
    CHAIRMAN_MODEL = _chairmen[0]
else:
    CHAIRMAN_MODEL = COUNCIL_MODELS[0]

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
