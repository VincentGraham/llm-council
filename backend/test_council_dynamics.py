"""Unit tests for dynamic deliberation helpers."""

from __future__ import annotations

import unittest

from backend.council import (
    _round_consensus_ratio,
    _should_early_stop,
    _synthesis_similarity,
)


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


if __name__ == "__main__":
    unittest.main()
