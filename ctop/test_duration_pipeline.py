"""Unit tests for CTOP duration prediction pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

# Support direct execution: `python -m unittest ctop/test_duration_pipeline.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .run_duration_pipeline import (
        LONG_OUTPUT_COLUMNS,
        _build_trial_text,
        _build_wide_predictions_df,
        _clean_text,
        execute_pipeline,
        run_duration_pipeline,
    )
except ImportError:
    from ctop.run_duration_pipeline import (
        LONG_OUTPUT_COLUMNS,
        _build_trial_text,
        _build_wide_predictions_df,
        _clean_text,
        execute_pipeline,
        run_duration_pipeline,
    )


def _input_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nct_id": "NCT00000001",
                "brief_title": "Trial A",
                "summary_text": "Summary text.",
                "description_text": "Description text.",
                "primary_outcomes_text": "Primary outcome text.",
                "secondary_outcomes_text": "Secondary outcome text.",
                "enrollment_count": 150,
                "enrollment_type": "Actual",
                "phase": "Phase 2",
                "overall_status": "Completed",
                "conditions_text": "Condition A | Condition B",
                "interventions_text": "Drug | Drug A",
                "duration_days": 304,
                "duration_months": 9.99,
                "duration_source": "primary_completion_date",
                "_start_date": "2020-01-01",
                "_completion_date": "2020-10-31",
            }
        ]
    )


def _baseline_result(evidence_count: int = 2) -> dict:
    evidence = [
        {"evidence_id": f"s-r3-chair-ev-{index:02d}", "maskable": True}
        for index in range(1, evidence_count + 1)
    ]
    return {
        "batch_id": "baseline-batch",
        "prompt_id": "baseline-prompt",
        "synthesis": {
            "status": "ok",
            "response": "Baseline response",
            "structured_parse_status": "parsed",
            "prediction": {
                "task_type": "duration",
                "value_numeric": 10.0,
                "value_text": "10 months",
                "unit": "months",
                "probability": 0.67,
                "label": "moderate",
                "confidence": 0.81,
            },
            "evidence": evidence,
        },
    }


def _cf_result(batch_id: str, prompt_id: str, numeric_value: float) -> dict:
    return {
        "batch_id": batch_id,
        "prompt_id": prompt_id,
        "synthesis": {
            "status": "ok",
            "response": "Counterfactual response",
            "structured_parse_status": "parsed",
            "prediction": {
                "task_type": "duration",
                "value_numeric": numeric_value,
                "value_text": f"{numeric_value} months",
                "unit": "months",
                "probability": 0.55,
                "label": "cf",
                "confidence": 0.7,
            },
            "evidence": [],
        },
    }


class DurationPipelineTest(unittest.IsolatedAsyncioTestCase):
    """Coverage for baseline/counterfactual orchestration and outputs."""

    async def test_baseline_success_with_per_evidence_counterfactual_fanout(self) -> None:
        input_df = _input_dataframe()

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=_baseline_result(evidence_count=2)),
        ) as baseline_mock, patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            new=AsyncMock(
                side_effect=[
                    _cf_result("cf-batch-1", "cf-prompt-1", 8.0),
                    _cf_result("cf-batch-2", "cf-prompt-2", 12.0),
                ]
            ),
        ) as cf_mock:
            long_df, wide_df, metrics = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        self.assertEqual(baseline_mock.await_count, 1)
        self.assertEqual(cf_mock.await_count, 2)
        self.assertEqual(len(long_df), 3)
        self.assertEqual(set(long_df["run_type"]), {"baseline", "counterfactual"})
        cf_ids = set(long_df[long_df["run_type"] == "counterfactual"]["masked_evidence_id"])
        self.assertEqual(cf_ids, {"s-r3-chair-ev-01", "s-r3-chair-ev-02"})

        self.assertEqual(len(wide_df), 1)
        self.assertEqual(int(wide_df.iloc[0]["cf_total"]), 2)
        self.assertEqual(int(wide_df.iloc[0]["cf_success"]), 2)
        self.assertEqual(int(wide_df.iloc[0]["cf_failed"]), 0)
        self.assertAlmostEqual(float(wide_df.iloc[0]["cf_delta_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["ema_hit_rate_epsilon"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["hit_rate_epsilon"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["mae"]), 0.01, places=6)

    async def test_baseline_failure_yields_error_row_and_no_counterfactuals(self) -> None:
        input_df = _input_dataframe()

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(side_effect=RuntimeError("baseline failed")),
        ), patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            new=AsyncMock(),
        ) as cf_mock:
            long_df, wide_df, metrics = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        self.assertEqual(cf_mock.await_count, 0)
        self.assertEqual(len(long_df), 1)
        row = long_df.iloc[0]
        self.assertEqual(row["run_type"], "baseline")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertEqual(len(wide_df), 1)
        self.assertEqual(int(wide_df.iloc[0]["cf_total"]), 0)
        self.assertEqual(int(metrics["evaluated_trials"]), 0)

    async def test_counterfactual_failure_rows_are_retained(self) -> None:
        input_df = _input_dataframe()

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=_baseline_result(evidence_count=2)),
        ), patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            new=AsyncMock(
                side_effect=[
                    RuntimeError("counterfactual failed"),
                    _cf_result("cf-batch-2", "cf-prompt-2", 11.0),
                ]
            ),
        ):
            long_df, wide_df, metrics = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        cf_rows = long_df[long_df["run_type"] == "counterfactual"].sort_values("cf_index")
        self.assertEqual(len(cf_rows), 2)
        self.assertEqual(cf_rows.iloc[0]["status"], "error")
        self.assertEqual(cf_rows.iloc[0]["error_type"], "RuntimeError")
        self.assertEqual(cf_rows.iloc[1]["status"], "ok")

        self.assertEqual(int(wide_df.iloc[0]["cf_total"]), 2)
        self.assertEqual(int(wide_df.iloc[0]["cf_success"]), 1)
        self.assertEqual(int(wide_df.iloc[0]["cf_failed"]), 1)
        self.assertEqual(int(metrics["evaluated_trials"]), 1)

    def test_build_trial_text_excludes_leakage_fields(self) -> None:
        row = _input_dataframe().iloc[0].to_dict()
        row["start_date"] = "January 2020"
        row["primary_completion_date"] = "October 2020"

        trial_text = _build_trial_text(row)

        self.assertIn("Brief title: Trial A", trial_text)
        self.assertIn("Enrollment: 150, type=Actual", trial_text)
        self.assertNotIn("duration_days", trial_text)
        self.assertNotIn("9.99", trial_text)
        self.assertNotIn("January 2020", trial_text)
        self.assertNotIn("October 2020", trial_text)

    def test_clean_text_handles_array_like_values(self) -> None:
        self.assertEqual(_clean_text([" months ", "days"]), "months")
        self.assertEqual(_clean_text(pd.array([" months ", None], dtype="string")), "months")
        self.assertIsNone(_clean_text(pd.array([None, pd.NA], dtype="object")))

    async def test_long_columns_and_wide_aggregates(self) -> None:
        input_df = _input_dataframe()
        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=_baseline_result(evidence_count=1)),
        ), patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            new=AsyncMock(return_value=_cf_result("cf-batch", "cf-prompt", 7.5)),
        ):
            long_df, wide_df, metrics = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        self.assertEqual(list(long_df.columns), LONG_OUTPUT_COLUMNS)
        self.assertEqual(len(long_df), 2)
        self.assertEqual(float(wide_df.iloc[0]["cf_numeric_mean"]), 7.5)
        self.assertEqual(float(wide_df.iloc[0]["cf_delta_mean"]), -2.5)
        self.assertIsNone(metrics["c_index"])

    async def test_execute_pipeline_writes_outputs_and_manifest(self) -> None:
        input_df = _input_dataframe()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.pkl"
            input_df.to_pickle(input_path)

            args = argparse.Namespace(
                input_path=input_path,
                output_root=tmp_path,
                run_id="abcd1234",
                prediction_target="duration",
                rounds=3,
                counterfactual=True,
                counterfactual_max_concurrency=2,
                allow_fuzzy_quotes=False,
                max_trials=None,
                epsilon_months=1.0,
                ema_alpha=0.12,
                short_threshold_months=12.0,
                progress=False,
                checkpoint_every_trials=1,
                strict=True,
            )

            with patch(
                "ctop.run_duration_pipeline.run_deliberation",
                new=AsyncMock(return_value=_baseline_result(evidence_count=1)),
            ), patch(
                "ctop.run_duration_pipeline.run_counterfactual_deliberation",
                new=AsyncMock(return_value=_cf_result("cf-batch", "cf-prompt", 9.0)),
            ), patch.object(
                pd.DataFrame,
                "to_parquet",
                autospec=True,
                side_effect=lambda self, path, index=False: Path(path).write_text(
                    "parquet",
                    encoding="utf-8",
                ),
            ):
                manifest = await execute_pipeline(args)

            run_dir = tmp_path / "abcd1234"
            expected_files = [
                "duration_predictions_long.csv",
                "duration_predictions_long.parquet",
                "duration_predictions_long.pkl",
                "duration_predictions_wide.csv",
                "duration_predictions_wide.parquet",
                "duration_predictions_wide.pkl",
                "run_manifest.json",
            ]
            for name in expected_files:
                self.assertTrue((run_dir / name).exists(), msg=f"missing output file {name}")

            checkpoint_files = [
                "duration_predictions_long.checkpoint.csv",
                "duration_predictions_long.checkpoint.parquet",
                "duration_predictions_long.checkpoint.pkl",
                "duration_predictions_wide.checkpoint.csv",
                "duration_predictions_wide.checkpoint.parquet",
                "duration_predictions_wide.checkpoint.pkl",
                "run_manifest.checkpoint.json",
            ]
            for name in checkpoint_files:
                self.assertTrue((run_dir / name).exists(), msg=f"missing checkpoint file {name}")

            manifest_path = run_dir / "run_manifest.json"
            persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "abcd1234")
            self.assertEqual(persisted_manifest["run_id"], "abcd1234")
            self.assertEqual(int(persisted_manifest["counterfactual_rows"]), 1)
            self.assertIn("evaluation_metrics", persisted_manifest)

    def test_build_wide_predictions_df_empty(self) -> None:
        wide_df = _build_wide_predictions_df(pd.DataFrame(columns=LONG_OUTPUT_COLUMNS))
        self.assertEqual(len(wide_df), 0)

    async def test_run_duration_pipeline_checkpoint_callback(self) -> None:
        input_df = _input_dataframe()
        snapshots: list[tuple[int, int]] = []

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=_baseline_result(evidence_count=0)),
        ):
            await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=False,
                allow_fuzzy_quotes=False,
                show_progress=False,
                checkpoint_every_trials=1,
                checkpoint_callback=lambda trial_index, rows_snapshot, _metrics: snapshots.append(
                    (trial_index, len(rows_snapshot))
                ),
            )

        self.assertEqual(snapshots, [(1, 1)])

    async def test_unmaskable_counterfactual_evidence_is_skipped(self) -> None:
        input_df = _input_dataframe()
        baseline_result = _baseline_result(evidence_count=2)
        baseline_result["synthesis"]["evidence"] = [
            {"evidence_id": "s-r3-chair-ev-01", "maskable": False},
            {"evidence_id": "s-r3-chair-ev-02", "maskable": False},
        ]

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=baseline_result),
        ), patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            new=AsyncMock(),
        ) as cf_mock:
            long_df, wide_df, _ = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=2,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        self.assertEqual(cf_mock.await_count, 0)
        self.assertEqual(len(long_df), 1)
        self.assertEqual(int(wide_df.iloc[0]["cf_total"]), 0)

    async def test_counterfactuals_respect_bounded_concurrency(self) -> None:
        input_df = _input_dataframe()
        in_flight = 0
        max_in_flight = 0
        call_index = 0

        async def fake_counterfactual(**_: object) -> dict:
            nonlocal in_flight, max_in_flight, call_index
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            call_index += 1
            in_flight -= 1
            return _cf_result(f"cf-batch-{call_index}", f"cf-prompt-{call_index}", 9.0)

        with patch(
            "ctop.run_duration_pipeline.run_deliberation",
            new=AsyncMock(return_value=_baseline_result(evidence_count=3)),
        ), patch(
            "ctop.run_duration_pipeline.run_counterfactual_deliberation",
            side_effect=fake_counterfactual,
        ):
            long_df, _, _ = await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=True,
                allow_fuzzy_quotes=False,
                counterfactual_max_concurrency=2,
                show_progress=False,
            )

        self.assertEqual(len(long_df[long_df["run_type"] == "counterfactual"]), 3)
        self.assertEqual(max_in_flight, 2)

    async def test_progress_reconcile_fills_missing_units(self) -> None:
        input_df = _input_dataframe()

        class FakeProgressReporter:
            latest_instance = None

            def __init__(self, total: int, enabled: bool = True) -> None:
                self.total = total
                self.enabled = enabled
                self.units = 0
                FakeProgressReporter.latest_instance = self

            def extend_total(self, units: int) -> None:
                self.total += units

            def advance_round(self, latest_line: str, metrics: object, units: int = 1) -> None:
                self.units += units

            def close(self) -> None:
                return

        baseline_error_row = {
            "nct_id": "NCT00000001",
            "run_type": "baseline",
            "status": "error",
            "error_message": "baseline failed",
            "prediction_value_numeric": None,
            "true_duration_months": 9.99,
        }

        with patch(
            "ctop.run_duration_pipeline._run_trial",
            new=AsyncMock(return_value=[baseline_error_row]),
        ), patch(
            "ctop.run_duration_pipeline.ProgressReporter",
            new=FakeProgressReporter,
        ):
            await run_duration_pipeline(
                input_df=input_df,
                run_id="run12345",
                prediction_target="duration",
                rounds=3,
                counterfactual_enabled=False,
                allow_fuzzy_quotes=False,
                show_progress=True,
            )

        reporter = FakeProgressReporter.latest_instance
        self.assertIsNotNone(reporter)
        self.assertEqual(reporter.units, reporter.total)


if __name__ == "__main__":
    unittest.main()
