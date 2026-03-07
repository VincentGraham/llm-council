"""Unit tests for evidence utility functions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

# Support direct script execution: `uv run backend/test_evidence.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .evidence import (
        extract_json_block_deterministic,
        mask_source_text,
        parse_hybrid_output,
        select_evidence_items,
    )
except ImportError:
    from backend.evidence import (
        extract_json_block_deterministic,
        mask_source_text,
        parse_hybrid_output,
        select_evidence_items,
    )


class EvidenceUtilsTest(unittest.TestCase):
    """Coverage for deterministic parsing, selection, and masking behavior."""

    def test_extract_json_block_deterministic(self) -> None:
        text = 'Answer\\n```json\\n{"prediction":{"label":"success"},"evidence":[]}\\n```'
        parsed = extract_json_block_deterministic(text)
        self.assertEqual(parsed["prediction"]["label"], "success")

    def test_parse_hybrid_output_exact_anchor(self) -> None:
        raw = """Narrative.\n```json\n{\n  "prediction": {"task_type": "success", "probability": 0.6},\n  "evidence": [\n    {"quote": "Primary endpoint met", "rationale": "efficacy", "confidence": 0.8},\n    {"quote": "sample size was 850", "rationale": "precision", "confidence": 0.7},\n    {"quote": "double blind", "rationale": "bias control", "confidence": 0.6}\n  ]\n}\n```"""
        source = "Primary endpoint met in phase 3. sample size was 850. It was double blind."
        parsed = asyncio.run(
            parse_hybrid_output(
                raw_text=raw,
                source_text=source,
                evidence_id_prefix="t1",
                allow_fuzzy_quotes=False,
            )
        )
        self.assertEqual(parsed["structured_parse_status"], "parsed")
        self.assertEqual(len(parsed["evidence"]), 3)
        self.assertTrue(all(item["match_type"] == "exact" for item in parsed["evidence"]))

    def test_parse_hybrid_output_fuzzy_anchor(self) -> None:
        raw = """Narrative.\n```json\n{\n  "prediction": {"task_type": "duration", "value_text": "12 months"},\n  "evidence": [\n    {"quote": "The trial reported median progression free survival of 12 months in a phase 3 randomized double-blind design", "rationale": "matches endpoint", "confidence": 0.7},\n    {"quote": "phase 3 randomized double-blind design", "rationale": "quality", "confidence": 0.6},\n    {"quote": "double blinded methods", "rationale": "low bias", "confidence": 0.6}\n  ]\n}\n```"""
        source = "The trial reported median progression-free survival of 12 months in a phase 3 randomized double blind design."
        parsed = asyncio.run(
            parse_hybrid_output(
                raw_text=raw,
                source_text=source,
                evidence_id_prefix="t2",
                allow_fuzzy_quotes=True,
            )
        )
        self.assertEqual(len(parsed["evidence"]), 3)
        self.assertTrue(any(item["match_type"] == "fuzzy" for item in parsed["evidence"]))

    def test_parse_hybrid_output_fuzzy_anchor_multi_sentence_quote(self) -> None:
        raw = """Narrative.\n```json\n{\n  "prediction": {"task_type": "success", "label": "likely"},\n  "evidence": [\n    {"quote": "Primary endpoint improved at week 24 and secondary endpoint improved by week 36.", "rationale": "cross-endpoint benefit", "confidence": 0.82},\n    {"quote": "randomized double blind", "rationale": "study quality", "confidence": 0.7},\n    {"quote": "n equals 1200", "rationale": "sample size", "confidence": 0.68}\n  ]\n}\n```"""
        source = (
            "Primary endpoint improved at week 24. Secondary endpoint improved by week 36. "
            "The trial was randomized double blind with n equals 1200 participants."
        )
        parsed = asyncio.run(
            parse_hybrid_output(
                raw_text=raw,
                source_text=source,
                evidence_id_prefix="t3",
                allow_fuzzy_quotes=True,
            )
        )
        self.assertEqual(len(parsed["evidence"]), 3)
        self.assertTrue(parsed["evidence"][0]["maskable"])
        self.assertEqual(parsed["evidence"][0]["match_type"], "fuzzy")

    def test_parse_hybrid_output_normalizes_scores_to_unit_interval(self) -> None:
        raw = """Narrative.\n```json\n{\n  "prediction": {"task_type": "duration", "probability": 80, "confidence": 6},\n  "evidence": [\n    {"quote": "Primary endpoint met", "rationale": "efficacy", "confidence": 9},\n    {"quote": "sample size was 850", "rationale": "precision", "confidence": 70},\n    {"quote": "double blind", "rationale": "bias control", "confidence": 0.6}\n  ]\n}\n```"""
        source = "Primary endpoint met in phase 3. sample size was 850. It was double blind."
        parsed = asyncio.run(
            parse_hybrid_output(
                raw_text=raw,
                source_text=source,
                evidence_id_prefix="t4",
                allow_fuzzy_quotes=False,
            )
        )
        self.assertEqual(parsed["prediction"]["probability"], 0.8)
        self.assertEqual(parsed["prediction"]["confidence"], 0.6)
        self.assertEqual(parsed["evidence"][0]["confidence"], 0.9)
        self.assertEqual(parsed["evidence"][1]["confidence"], 0.7)

    def test_select_evidence_union(self) -> None:
        evidence_index = [
            {
                "evidence_id": "a",
                "model": "m1",
                "round": 1,
                "source_tag": "design",
                "is_synthesis": False,
            },
            {
                "evidence_id": "b",
                "model": "m2",
                "round": 2,
                "source_tag": "endpoint",
                "is_synthesis": False,
            },
        ]
        selected = select_evidence_items(
            evidence_index,
            evidence_ids=["a"],
            selectors={"models": ["m2"]},
        )
        selected_ids = {item["evidence_id"] for item in selected}
        self.assertEqual(selected_ids, {"a", "b"})

    def test_mask_source_text_overlap(self) -> None:
        source = "The primary endpoint met and endpoint met strongly."
        selected = [
            {"evidence_id": "a", "span_start": 4, "span_end": 24, "maskable": True},
            {"evidence_id": "b", "span_start": 16, "span_end": 31, "maskable": True},
        ]
        masked_text, manifest, masked_ids = mask_source_text(source, selected)
        self.assertIn("[EVIDENCE_001]", masked_text)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(set(masked_ids), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
