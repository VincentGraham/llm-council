"""Unit tests for dynamic deliberation helpers."""

from __future__ import annotations

import unittest

from backend.council import (
    _build_round_n_prompt,
    _primary_prompt_from_request,
    _prediction_signature,
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

    def test_prediction_signature_includes_probability_with_label(self) -> None:
        sig_a = _prediction_signature({"label": "success", "probability": 0.9})
        sig_b = _prediction_signature({"label": "success", "probability": 0.3})
        self.assertNotEqual(sig_a, sig_b)

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

    def test_invalid_inference_overrides_log_warning(self) -> None:
        payload = {"inference": {"round1": {"temperature": "not-a-number", "max_tokens": -5, "foo": 1}}}
        with self.assertLogs("backend.council", level="WARNING") as logs:
            resolved = _resolve_stage_inference(payload, "round1", ROUND1_INFERENCE_PARAMS)
        self.assertEqual(resolved, ROUND1_INFERENCE_PARAMS)
        self.assertTrue(any("Ignoring" in line for line in logs.output))

    def test_share_synthesis_with_members_toggle(self) -> None:
        prior_rounds = [{"round": 1, "responses": [{"model": "m2", "response": "response text"}]}]
        prior_syntheses = [{"synthesis": {"response": "synthesis text"}}]

        prompt_hidden, _ = _build_round_n_prompt(
            model_name="m1",
            request_payload={"prompt": "task"},
            prior_rounds=prior_rounds,
            prior_syntheses=prior_syntheses,
            share_synthesis_with_members=False,
        )
        self.assertNotIn("Latest chairman synthesis", prompt_hidden)

        prompt_shared, _ = _build_round_n_prompt(
            model_name="m1",
            request_payload={"prompt": "task"},
            prior_rounds=prior_rounds,
            prior_syntheses=prior_syntheses,
            share_synthesis_with_members=True,
        )
        self.assertIn("Latest chairman synthesis", prompt_shared)

    def test_round_n_labels_remain_stable_when_model_missing(self) -> None:
        prior_rounds = [
            {
                "round": 1,
                "responses": [
                    {"model": "m2", "response": "first m2"},
                    {"model": "m3", "response": "first m3"},
                ],
            },
            {
                "round": 2,
                "responses": [
                    {"model": "m2", "response": ""},
                    {"model": "m3", "response": "second m3"},
                ],
            },
        ]

        _, latest_map = _build_round_n_prompt(
            model_name="m1",
            request_payload={"prompt": "task"},
            prior_rounds=prior_rounds,
            prior_syntheses=None,
            share_synthesis_with_members=False,
            council_members=["m1", "m2", "m3"],
        )
        self.assertEqual(latest_map, {"Response B": "m3"})

    def test_primary_prompt_duration_target_guidance(self) -> None:
        prompt = _primary_prompt_from_request(
            {
                "trial_text": "Study text",
                "prediction_target": "duration",
            }
        )
        self.assertIn("TOTAL trial duration", prompt)
        self.assertIn("primary endpoint", prompt)
        self.assertIn("Do not predict intervention/treatment duration", prompt)
        self.assertIn("specific trial or protocol", prompt)
        self.assertIn("not the lifespan of a broader network", prompt)

    def test_primary_prompt_non_duration_target_has_no_duration_guidance(self) -> None:
        prompt = _primary_prompt_from_request(
            {
                "trial_text": "Study text",
                "prediction_target": "response_rate",
            }
        )
        self.assertNotIn("TOTAL trial duration", prompt)


if __name__ == "__main__":
    unittest.main()
