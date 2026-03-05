"""Unit tests for models.yaml schema validation."""

from __future__ import annotations

import unittest

from backend.config_schema import ModelConfig


class ConfigSchemaTest(unittest.TestCase):
    """Coverage for basic model field validation."""

    def test_rejects_invalid_gpu_format(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(
                name="m1",
                image="nvcr.io/nim/meta/llama:latest",
                gpus="a,b",
                port=8000,
            )

    def test_rejects_invalid_image_format(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(
                name="m1",
                image="invalid image",
                gpus="0,1",
                port=8000,
            )


if __name__ == "__main__":
    unittest.main()
