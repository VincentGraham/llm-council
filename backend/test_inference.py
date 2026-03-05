"""Unit tests for inference helpers."""

from __future__ import annotations

import unittest

from backend.inference import _match_requested_model


class InferenceHelpersTest(unittest.TestCase):
    """Coverage for request-model health matching behavior."""

    def test_match_requested_model_exact(self) -> None:
        ok, matched, match_type = _match_requested_model(
            "meta/llama-3.1-70b-instruct",
            ["meta/llama-3.1-70b-instruct"],
        )
        self.assertTrue(ok)
        self.assertEqual(matched, "meta/llama-3.1-70b-instruct")
        self.assertEqual(match_type, "exact")

    def test_match_requested_model_normalized(self) -> None:
        ok, matched, match_type = _match_requested_model(
            "meta/llama-3.1-70b-instruct",
            ["meta-llama-3.1-70b-instruct"],
        )
        self.assertTrue(ok)
        self.assertEqual(matched, "meta-llama-3.1-70b-instruct")
        self.assertEqual(match_type, "normalized")

    def test_match_requested_model_missing(self) -> None:
        ok, matched, match_type = _match_requested_model(
            "meta/llama-3.1-70b-instruct",
            ["qwen/qwen2.5-72b-instruct"],
        )
        self.assertFalse(ok)
        self.assertIsNone(matched)
        self.assertEqual(match_type, "missing")


if __name__ == "__main__":
    unittest.main()
