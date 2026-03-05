"""Unit tests for CTOP XML parsing and dataframe pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

# Support direct execution: `uv run python ctop/test_trial_xml_pipeline.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .build_trials_dataframe import (
        build_trials_dataframe,
        write_trials_dataframe,
    )
    from .trial_xml_parser import parse_clinical_date, parse_trial_xml
except ImportError:
    from ctop.build_trials_dataframe import build_trials_dataframe, write_trials_dataframe
    from ctop.trial_xml_parser import parse_clinical_date, parse_trial_xml


XML_WITH_PRIMARY = """\
<clinical_study>
  <id_info><nct_id>NCT00000001</nct_id></id_info>
  <brief_title>Study With Primary Completion</brief_title>
  <brief_summary><textblock>Brief summary for study one.</textblock></brief_summary>
  <detailed_description><textblock>Detailed description for study one.</textblock></detailed_description>
  <overall_status>Completed</overall_status>
  <phase>Phase 2</phase>
  <enrollment type="Actual">120</enrollment>
  <condition>Condition A</condition>
  <condition>Condition B</condition>
  <intervention>
    <intervention_type>Drug</intervention_type>
    <intervention_name>Drug A</intervention_name>
    <description>Main treatment</description>
  </intervention>
  <primary_outcome>
    <measure>Progression-free survival</measure>
    <description>Measured by imaging.</description>
    <time_frame>12 months</time_frame>
  </primary_outcome>
  <secondary_outcome>
    <measure>Overall survival</measure>
    <time_frame>24 months</time_frame>
  </secondary_outcome>
  <start_date>January 2020</start_date>
  <primary_completion_date>March 2020</primary_completion_date>
  <completion_date>April 2020</completion_date>
</clinical_study>
"""

XML_WITH_COMPLETION_FALLBACK = """\
<clinical_study>
  <id_info><nct_id>NCT00000002</nct_id></id_info>
  <brief_title>Study With Completion Fallback</brief_title>
  <brief_summary><textblock>Brief summary for study two.</textblock></brief_summary>
  <detailed_description><textblock>Detailed description for study two.</textblock></detailed_description>
  <overall_status>Completed</overall_status>
  <phase>Phase 3</phase>
  <enrollment type="Anticipated">300</enrollment>
  <condition>Condition C</condition>
  <intervention>
    <intervention_type>Biological</intervention_type>
    <intervention_name>Therapy B</intervention_name>
  </intervention>
  <primary_outcome>
    <measure>Response rate</measure>
    <time_frame>6 months</time_frame>
  </primary_outcome>
  <start_date>January 1, 2020</start_date>
  <completion_date>2020-04-01</completion_date>
</clinical_study>
"""

XML_MISSING_DURATION = """\
<clinical_study>
  <id_info><nct_id>NCT00000003</nct_id></id_info>
  <brief_title>Study Missing Duration</brief_title>
  <brief_summary><textblock>Brief summary for study three.</textblock></brief_summary>
  <overall_status>Recruiting</overall_status>
  <phase>Phase 1</phase>
  <enrollment type="Actual">90</enrollment>
  <condition>Condition D</condition>
  <completion_date>2021-06-01</completion_date>
</clinical_study>
"""


class TrialXmlPipelineTest(unittest.TestCase):
    """Coverage for CTOP parser and dataframe builder behavior."""

    def test_extracts_semantic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "NCT00000001.xml"
            xml_path.write_text(XML_WITH_PRIMARY, encoding="utf-8")

            parsed = parse_trial_xml(xml_path)

            self.assertEqual(parsed["nct_id"], "NCT00000001")
            self.assertEqual(parsed["enrollment_count"], 120)
            self.assertEqual(parsed["enrollment_type"], "Actual")
            self.assertEqual(parsed["phase"], "Phase 2")
            self.assertEqual(parsed["overall_status"], "Completed")
            self.assertIn("Brief summary", parsed["summary_text"] or "")
            self.assertIn("measure: Progression-free survival", parsed["primary_outcomes_text"] or "")
            self.assertIn("Condition A", parsed["conditions_text"] or "")
            self.assertIn("Drug A", parsed["interventions_text"] or "")

    def test_uses_primary_completion_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "NCT00000001.xml").write_text(XML_WITH_PRIMARY, encoding="utf-8")

            dataframe, stats, _ = build_trials_dataframe(input_glob=f"{tmpdir}/*.xml")

            self.assertEqual(stats["total_files"], 1)
            self.assertEqual(len(dataframe), 1)
            row = dataframe.iloc[0]
            self.assertEqual(row["duration_source"], "primary_completion_date")
            self.assertEqual(int(row["duration_days"]), 60)
            self.assertEqual(float(row["duration_months"]), 1.97)

    def test_falls_back_to_completion_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "NCT00000002.xml").write_text(XML_WITH_COMPLETION_FALLBACK, encoding="utf-8")

            dataframe, _, _ = build_trials_dataframe(input_glob=f"{tmpdir}/*.xml")

            self.assertEqual(len(dataframe), 1)
            row = dataframe.iloc[0]
            self.assertEqual(row["duration_source"], "completion_date")
            self.assertEqual(int(row["duration_days"]), 91)
            self.assertEqual(float(row["duration_months"]), 2.99)

    def test_month_year_dates_default_to_first_of_month(self) -> None:
        self.assertEqual(parse_clinical_date("January 2020"), date(2020, 1, 1))
        self.assertEqual(parse_clinical_date("Mar 2020"), date(2020, 3, 1))

    def test_drops_rows_without_computable_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "NCT00000001.xml").write_text(XML_WITH_PRIMARY, encoding="utf-8")
            (Path(tmpdir) / "NCT00000003.xml").write_text(XML_MISSING_DURATION, encoding="utf-8")

            dataframe, stats, failures = build_trials_dataframe(input_glob=f"{tmpdir}/*.xml")

            self.assertEqual(stats["total_files"], 2)
            self.assertEqual(stats["labeled_trials"], 1)
            self.assertEqual(stats["dropped_trials"], 1)
            self.assertEqual(len(dataframe), 1)
            self.assertEqual(len(failures), 1)
            self.assertIn("duration_uncomputable", failures[0]["reason"])

    def test_dataframe_excludes_raw_date_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "NCT00000001.xml").write_text(XML_WITH_PRIMARY, encoding="utf-8")

            dataframe, _, _ = build_trials_dataframe(input_glob=f"{tmpdir}/*.xml")

            self.assertNotIn("start_date", dataframe.columns)
            self.assertNotIn("primary_completion_date", dataframe.columns)
            self.assertNotIn("completion_date", dataframe.columns)
            self.assertNotIn("_start_date", dataframe.columns)
            self.assertNotIn("_primary_completion_date", dataframe.columns)
            self.assertNotIn("_completion_date", dataframe.columns)

    def test_strict_mode_fails_when_output_path_is_not_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_as_file = Path(tmpdir) / "not_a_directory"
            output_as_file.write_text("content", encoding="utf-8")
            dataframe = pd.DataFrame([{"nct_id": "NCTX", "duration_days": 1, "duration_months": 0.03}])

            with self.assertRaises(NotADirectoryError):
                write_trials_dataframe(dataframe=dataframe, output_dir=output_as_file, strict=True)

    def test_writes_parquet_and_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "NCT00000001.xml").write_text(XML_WITH_PRIMARY, encoding="utf-8")
            dataframe, _, _ = build_trials_dataframe(input_glob=f"{tmpdir}/*.xml")

            output_dir = Path(tmpdir) / "out"
            try:
                parquet_path, csv_path = write_trials_dataframe(
                    dataframe=dataframe,
                    output_dir=output_dir,
                    strict=True,
                )
            except RuntimeError as exc:
                if "parquet engine" in str(exc).lower():
                    self.skipTest(f"Parquet engine unavailable in test environment: {exc}")
                raise

            self.assertTrue(parquet_path.exists())
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
