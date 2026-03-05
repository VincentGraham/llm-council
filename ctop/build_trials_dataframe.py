#!/usr/bin/env python3
"""Build an intermediate CTOP dataframe from ClinicalTrials.gov XML trials."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

try:
    from .trial_xml_parser import derive_duration_label, parse_trial_xml
except ImportError:
    from ctop.trial_xml_parser import derive_duration_label, parse_trial_xml


logger = logging.getLogger(__name__)

DEFAULT_INPUT_GLOB = "data/trials/TOP/*.xml"
DEFAULT_OUTPUT_DIR = Path("/data/trials/output")

OUTPUT_COLUMNS = [
    "nct_id",
    "xml_nct_id",
    "nct_id_mismatch",
    "brief_title",
    "summary_text",
    "description_text",
    "primary_outcomes_text",
    "secondary_outcomes_text",
    "enrollment_count",
    "enrollment_type",
    "phase",
    "overall_status",
    "conditions_text",
    "interventions_text",
    "duration_days",
    "duration_months",
    "duration_source",
]


def _resolve_input_files(input_glob: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(input_glob))


def build_trials_dataframe(
    input_glob: str = DEFAULT_INPUT_GLOB,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, str]]]:
    """
    Build a dataframe of trials with semantic sections and duration labels.

    Trials without computable duration labels are dropped.
    """
    xml_paths = _resolve_input_files(input_glob)
    if not xml_paths:
        raise FileNotFoundError(f"No XML files matched input glob: {input_glob}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    parsed_trials = 0
    dropped_trials = 0

    for xml_path in xml_paths:
        try:
            parsed = parse_trial_xml(xml_path)
            parsed_trials += 1
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(
                {
                    "nct_id": xml_path.stem,
                    "reason": f"xml_parse_error: {exc}",
                    "path": str(xml_path),
                }
            )
            continue

        duration_label, duration_error = derive_duration_label(
            start_date=parsed.get("_start_date"),
            primary_completion_date=parsed.get("_primary_completion_date"),
            completion_date=parsed.get("_completion_date"),
        )
        if duration_label is None:
            dropped_trials += 1
            failures.append(
                {
                    "nct_id": str(parsed.get("nct_id") or xml_path.stem),
                    "reason": f"duration_uncomputable: {duration_error}",
                    "path": str(xml_path),
                }
            )
            continue

        row = {key: value for key, value in parsed.items() if not key.startswith("_")}
        row.update(duration_label)
        rows.append(row)

    dataframe = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    stats = {
        "total_files": len(xml_paths),
        "parsed_trials": parsed_trials,
        "labeled_trials": len(dataframe),
        "dropped_trials": dropped_trials,
        "parse_failures": len(xml_paths) - parsed_trials,
    }
    return dataframe, stats, failures


def ensure_output_dir(output_dir: Path, strict: bool = True) -> None:
    """Ensure output path exists and is writable."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Could not create output directory '{output_dir}': {exc}"
        if strict:
            raise RuntimeError(message) from exc
        logger.warning(message)
        return

    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, delete=True, encoding="utf-8") as handle:
            handle.write("ctop_write_check")
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Output directory '{output_dir}' is not writable: {exc}"
        if strict:
            raise PermissionError(message) from exc
        logger.warning(message)


def write_trials_dataframe(
    dataframe: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strict: bool = True,
) -> tuple[Path, Path]:
    """Write intermediate dataframe to parquet and CSV."""
    ensure_output_dir(output_dir=output_dir, strict=strict)

    parquet_path = output_dir / "trials_intermediate.parquet"
    csv_path = output_dir / "trials_intermediate.csv"

    try:
        dataframe.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "Failed to write parquet file. Install a parquet engine such as pyarrow."
        ) from exc
    dataframe.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help=f"Input XML glob pattern (default: {DEFAULT_INPUT_GLOB})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if output path is unavailable or not writable (default: true).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    dataframe, stats, failures = build_trials_dataframe(input_glob=args.input_glob)
    parquet_path, csv_path = write_trials_dataframe(
        dataframe=dataframe,
        output_dir=args.output_dir,
        strict=bool(args.strict),
    )

    logger.info("Processed XML files: %s", stats["total_files"])
    logger.info("Trials with duration labels: %s", stats["labeled_trials"])
    logger.info("Dropped trials: %s", stats["dropped_trials"])
    logger.info("Parse failures: %s", stats["parse_failures"])
    if failures:
        logger.info("Recorded failures: %s", len(failures))
    logger.info("Wrote parquet: %s", parquet_path)
    logger.info("Wrote csv: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
