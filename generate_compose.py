#!/usr/bin/env python3
"""Generate docker-compose.yml from models.yaml."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("models.yaml")
DEFAULT_OUTPUT_PATH = Path("docker-compose.yml")


def slugify_service_name(name: str) -> str:
    """Convert model name into a safe docker compose service name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "nim-model"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate models yaml config."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("models.yaml must be a mapping.")

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models.yaml must define a non-empty 'models' list.")

    required_fields = {"name", "image", "gpus", "port"}
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{index}] must be a mapping.")
        missing = required_fields - model.keys()
        if missing:
            raise ValueError(f"models[{index}] missing fields: {sorted(missing)}")

    return data


def warn_gpu_overlap(models: list[dict[str, Any]]) -> None:
    """Print a warning if GPU assignments overlap across models."""
    assigned: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []

    for model in models:
        model_name = str(model["name"])
        gpu_ids = [gpu.strip() for gpu in str(model["gpus"]).split(",") if gpu.strip()]
        for gpu_id in gpu_ids:
            if gpu_id in assigned:
                overlaps.append((gpu_id, assigned[gpu_id], model_name))
            else:
                assigned[gpu_id] = model_name

    if overlaps:
        print("Warning: overlapping GPU assignments detected:", file=sys.stderr)
        for gpu_id, first_model, second_model in overlaps:
            print(
                f"  GPU {gpu_id}: {first_model} and {second_model}",
                file=sys.stderr,
            )


def build_compose(config: dict[str, Any]) -> dict[str, Any]:
    """Build docker compose dictionary from model config."""
    services: dict[str, Any] = {}
    used_service_names: set[str] = set()

    models = config["models"]
    warn_gpu_overlap(models)

    for model in models:
        model_name = str(model["name"])
        service_name = slugify_service_name(model_name)
        while service_name in used_service_names:
            service_name = f"{service_name}-x"
        used_service_names.add(service_name)

        port = int(model["port"])
        gpus = str(model["gpus"])

        services[service_name] = {
            "image": str(model["image"]),
            "container_name": f"nim-{service_name}",
            "runtime": "nvidia",
            "ipc": "host",
            "ulimits": {"memlock": -1, "stack": 67108864},
            "environment": [
                "NGC_API_KEY=${NGC_API_KEY}",
                f"NVIDIA_VISIBLE_DEVICES={gpus}",
                "NIM_INFERENCE_BACKEND=vllm",
                "NIM_CACHE_PATH=/opt/nim/.cache",
                "HF_HOME=/opt/nim/.cache/huggingface",
                "TRANSFORMERS_CACHE=/opt/nim/.cache/huggingface",
            ],
            "volumes": ["/data/nim-cache:/opt/nim/.cache"],
            "ports": [f"{port}:8000"],
            "healthcheck": {
                "test": [
                    "CMD-SHELL",
                    "curl -fsS http://localhost:8000/v1/models || exit 1",
                ],
                "interval": "30s",
                "timeout": "10s",
                "retries": 40,
                "start_period": "120s",
            },
            "restart": "unless-stopped",
        }

    return {"version": "3.9", "services": services}


def write_compose(compose: dict[str, Any], path: Path) -> None:
    """Write docker compose yaml to output path."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(compose, handle, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to models.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output docker compose file path",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)
    compose = build_compose(config)
    write_compose(compose, args.output)
    print(f"Wrote compose file to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
