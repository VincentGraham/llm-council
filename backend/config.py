"""Configuration loader for local NIM/vLLM council."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML_PATH = Path(os.getenv("MODELS_YAML", PROJECT_ROOT / "models.yaml"))
DATA_DIR = os.getenv("DATA_DIR", "/data/results")
EVIDENCE_MIN_ITEMS = 3
EVIDENCE_MAX_ITEMS = 5


def _load_models_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {path}. "
            "Set MODELS_YAML or create models.yaml in the project root."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("models.yaml must be a mapping.")

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models.yaml must include a non-empty 'models' list.")

    rounds = data.get("rounds", 3)
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("'rounds' must be an integer >= 1.")
    extractor_model = data.get("extractor_model")
    if extractor_model is not None and not isinstance(extractor_model, str):
        raise ValueError("'extractor_model' must be a string when provided.")

    seen_names = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{index}] must be a mapping.")

        name = model.get("name")
        port = model.get("port")
        image = model.get("image")
        gpus = model.get("gpus")
        request_model = model.get("request_model")

        if not name or not isinstance(name, str):
            raise ValueError(f"models[{index}].name must be a non-empty string.")
        if name in seen_names:
            raise ValueError(f"Duplicate model name in models.yaml: {name}")
        seen_names.add(name)

        if not image or not isinstance(image, str):
            raise ValueError(f"models[{index}].image must be a non-empty string.")
        if gpus is None:
            raise ValueError(f"models[{index}].gpus is required.")
        if not isinstance(port, int):
            raise ValueError(f"models[{index}].port must be an integer.")
        if request_model is not None and not isinstance(request_model, str):
            raise ValueError(f"models[{index}].request_model must be a string.")

        model.setdefault("chairman", False)

    if extractor_model and extractor_model not in seen_names:
        raise ValueError("'extractor_model' must reference one of configured model names.")

    return {"rounds": rounds, "models": models, "extractor_model": extractor_model}


_MODELS_CONFIG = _load_models_config(MODELS_YAML_PATH)

MODELS = _MODELS_CONFIG["models"]
ROUNDS = _MODELS_CONFIG["rounds"]

COUNCIL_MODELS = [model["name"] for model in MODELS]
MODEL_CONFIG_BY_NAME = {model["name"]: model for model in MODELS}
MODEL_ENDPOINTS = {
    model["name"]: f"http://localhost:{model['port']}/v1"
    for model in MODELS
}
MODEL_REQUEST_NAMES = {
    model["name"]: model.get("request_model", model["name"])
    for model in MODELS
}

_chairmen = [model["name"] for model in MODELS if model.get("chairman")]
if len(_chairmen) > 1:
    raise ValueError("models.yaml can only have one chairman model.")
if _chairmen:
    CHAIRMAN_MODEL = _chairmen[0]
else:
    CHAIRMAN_MODEL = COUNCIL_MODELS[0]

EXTRACTOR_MODEL = _MODELS_CONFIG.get("extractor_model") or CHAIRMAN_MODEL
