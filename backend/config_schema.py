"""Shared models.yaml schema and loader for backend + tooling."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ModelConfig(BaseModel):
    """Per-model configuration in models.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    image: str
    gpus: str
    port: int
    request_model: str | None = None
    endpoint: str | None = None
    chairman: bool = False


class DeliberationConfig(BaseModel):
    """Optional deliberation controls for dynamic trajectories."""

    model_config = ConfigDict(extra="forbid")

    observer_chairman: bool = True
    share_synthesis_with_members: bool = False
    early_stopping: bool = True
    min_rounds_before_stop: int = 2
    synthesis_similarity_threshold: float = 0.985
    consensus_ratio_threshold: float = 0.8

    @model_validator(mode="after")
    def validate_ranges(self) -> "DeliberationConfig":
        if self.min_rounds_before_stop < 1:
            raise ValueError("'min_rounds_before_stop' must be >= 1")
        if not 0.0 <= self.synthesis_similarity_threshold <= 1.0:
            raise ValueError("'synthesis_similarity_threshold' must be between 0 and 1")
        if not 0.0 <= self.consensus_ratio_threshold <= 1.0:
            raise ValueError("'consensus_ratio_threshold' must be between 0 and 1")
        return self


class StageInferenceConfig(BaseModel):
    """Generation parameters for one deliberation stage."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "StageInferenceConfig":
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("'temperature' must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("'max_tokens' must be >= 1")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("'top_p' must be in (0, 1]")
        return self


class InferenceConfig(BaseModel):
    """Per-stage inference controls for council runs."""

    model_config = ConfigDict(extra="forbid")

    round1: StageInferenceConfig = Field(
        default_factory=lambda: StageInferenceConfig(temperature=0.7, max_tokens=2200)
    )
    round_n: StageInferenceConfig = Field(
        default_factory=lambda: StageInferenceConfig(temperature=0.5, max_tokens=2200)
    )
    synthesis: StageInferenceConfig = Field(
        default_factory=lambda: StageInferenceConfig(temperature=0.25, max_tokens=1800)
    )
    extractor: StageInferenceConfig = Field(
        default_factory=lambda: StageInferenceConfig(temperature=0.0, max_tokens=1200)
    )


class CouncilConfig(BaseModel):
    """Top-level models.yaml configuration."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = 3
    extractor_model: str | None = None
    deliberation: DeliberationConfig = Field(default_factory=DeliberationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
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


def load_council_config(path: Path) -> CouncilConfig:
    """Load and validate models.yaml from the provided path."""
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
