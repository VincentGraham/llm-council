"""Unit tests for dynamic deliberation helpers."""

from __future__ import annotations

import unittest

from backend.council import (
    _resolve_stage_inference,
    _round_consensus_ratio,
    _summarize_usage,
    _should_early_stop,
    _synthesis_similarity,
)
from backend.config import ROUND1_INFERENCE_PARAMS


class CouncilDynamicsTest(unittest.TestCase):
    """Coverage for early-stopping trajectory heuristics."""

    def test_round_consensus_ratio(self) -> None:
        responses = [
            {"prediction": {"label": "success"}},
            {"prediction": {"label": "success"}},
            {"prediction": {"label": "failure"}},
        ]
        ratio = _round_consensus_ratio(responses)
        self.assertAlmostEqual(ratio, 2 / 3, places=6)

    def test_synthesis_similarity(self) -> None:
        prev = "The likely duration is about 10 months."
        curr = "The likely duration is about 10 months with moderate confidence."
        similarity = _synthesis_similarity(prev, curr)
        self.assertIsNotNone(similarity)
        self.assertGreater(similarity, 0.7)

    def test_should_early_stop_on_consensus(self) -> None:
        stop, reason = _should_early_stop(
            round_number=2,
            early_stopping_enabled=True,
            min_rounds_before_stop=2,
            consensus_ratio=1.0,
            synthesis_similarity=0.2,
        )
        self.assertTrue(stop)
        self.assertIn("consensus_ratio", reason or "")

    def test_should_early_stop_on_similarity(self) -> None:
        stop, reason = _should_early_stop(
            round_number=2,
            early_stopping_enabled=True,
            min_rounds_before_stop=2,
            consensus_ratio=0.5,
            synthesis_similarity=0.99,
        )
        self.assertTrue(stop)
        self.assertIn("synthesis_similarity", reason or "")

    def test_should_not_stop_before_min_round(self) -> None:
        stop, _ = _should_early_stop(
            round_number=1,
            early_stopping_enabled=True,
            min_rounds_before_stop=2,
            consensus_ratio=1.0,
            synthesis_similarity=1.0,
        )
        self.assertFalse(stop)

    def test_resolve_stage_inference_overrides(self) -> None:
        payload = {"inference": {"round1": {"temperature": 0.9, "max_tokens": 999}}}
        resolved = _resolve_stage_inference(payload, "round1", ROUND1_INFERENCE_PARAMS)
        self.assertEqual(resolved["temperature"], 0.9)
        self.assertEqual(resolved["max_tokens"], 999)

    def test_summarize_usage(self) -> None:
        rounds = [
            {"responses": [{"model": "m1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]},
            {"responses": [{"model": "m2", "usage": {"prompt_tokens": 7, "completion_tokens": 8}}]},
        ]
        syntheses = [
            {"synthesis": {"model": "chair", "usage": {"prompt_tokens": 6, "completion_tokens": 4}}}
        ]
        summary = _summarize_usage(rounds, syntheses)
        self.assertEqual(summary["calls_with_usage"], 3)
        self.assertEqual(summary["totals"]["prompt_tokens"], 23)
        self.assertEqual(summary["totals"]["completion_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
