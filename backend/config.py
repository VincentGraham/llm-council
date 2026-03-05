"""Configuration loader for local NIM/vLLM council."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML_PATH = Path(os.getenv("MODELS_YAML", PROJECT_ROOT / "models.yaml"))
DATA_DIR = os.getenv("DATA_DIR", "/data/results")
EVIDENCE_MIN_ITEMS = 3
EVIDENCE_MAX_ITEMS = 5


class ModelConfig(BaseModel):
    """Per-model configuration in models.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    image: str
    gpus: str
    port: int
    request_model: str | None = None
    chairman: bool = False


class DeliberationConfig(BaseModel):
    """Optional deliberation controls for dynamic trajectories."""

    model_config = ConfigDict(extra="forbid")

    observer_chairman: bool = True
    early_stopping: bool = True
    min_rounds_before_stop: int = 2
    synthesis_similarity_threshold: float = 0.985
    consensus_ratio_threshold: float = 1.0

    @model_validator(mode="after")
    def validate_ranges(self) -> "DeliberationConfig":
        if self.min_rounds_before_stop < 1:
            raise ValueError("'min_rounds_before_stop' must be >= 1")
        if not 0.0 <= self.synthesis_similarity_threshold <= 1.0:
            raise ValueError("'synthesis_similarity_threshold' must be between 0 and 1")
        if not 0.0 <= self.consensus_ratio_threshold <= 1.0:
            raise ValueError("'consensus_ratio_threshold' must be between 0 and 1")
        return self


class CouncilConfig(BaseModel):
    """Top-level models.yaml configuration."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = 3
    extractor_model: str | None = None
    deliberation: DeliberationConfig = DeliberationConfig()
    models: list[ModelConfig]

    @model_validator(mode="after")
    def validate_consistency(self) -> "CouncilConfig":
        if self.rounds < 1:
            raise ValueError("'rounds' must be an integer >= 1.")
        if not self.models:
            raise ValueError("models.yaml must include a non-empty 'models' list.")

        names = [model.name for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError("Model names in models.yaml must be unique.")

        chairmen = [model.name for model in self.models if model.chairman]
        if len(chairmen) > 1:
            raise ValueError("models.yaml can only have one chairman model.")

        if self.extractor_model and self.extractor_model not in names:
            raise ValueError("'extractor_model' must reference one of configured model names.")

        return self


def _load_models_config(path: Path) -> CouncilConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {path}. "
            "Set MODELS_YAML or create models.yaml in the project root."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("models.yaml must be a mapping.")

    try:
        return CouncilConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ValueError(f"Invalid models.yaml configuration:\n{exc}") from exc


_MODELS_CONFIG = _load_models_config(MODELS_YAML_PATH)

MODELS = [model.model_dump() for model in _MODELS_CONFIG.models]
ROUNDS = _MODELS_CONFIG.rounds

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
if _chairmen:
    CHAIRMAN_MODEL = _chairmen[0]
else:
    CHAIRMAN_MODEL = COUNCIL_MODELS[0]

EXTRACTOR_MODEL = _MODELS_CONFIG.extractor_model or CHAIRMAN_MODEL

OBSERVER_CHAIRMAN_MODE = _MODELS_CONFIG.deliberation.observer_chairman
EARLY_STOP_ENABLED_DEFAULT = _MODELS_CONFIG.deliberation.early_stopping
EARLY_STOP_MIN_ROUNDS = _MODELS_CONFIG.deliberation.min_rounds_before_stop
SYNTHESIS_SIMILARITY_THRESHOLD = _MODELS_CONFIG.deliberation.synthesis_similarity_threshold
CONSENSUS_RATIO_THRESHOLD = _MODELS_CONFIG.deliberation.consensus_ratio_threshold
