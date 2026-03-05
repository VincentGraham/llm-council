"""Parse ClinicalTrials.gov classic XML files into semantic trial fields."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %Y",
    "%b %Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m",
    "%Y/%m",
    "%Y",
)

_EMPTY_DATE_TOKENS = {
    "",
    "n/a",
    "na",
    "none",
    "not available",
    "unknown",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child(node: ET.Element, child_name: str) -> ET.Element | None:
    for child in list(node):
        if _local_name(child.tag) == child_name:
            return child
    return None


def _find_children(node: ET.Element, child_name: str) -> list[ET.Element]:
    return [child for child in list(node) if _local_name(child.tag) == child_name]


def _find_path(node: ET.Element, path: str) -> ET.Element | None:
    current: ET.Element | None = node
    for part in path.split("/"):
        if current is None:
            return None
        current = _find_child(current, part)
    return current


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _node_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    return _clean_text("".join(node.itertext()))


def _joined_or_none(values: list[str], separator: str = " | ") -> str | None:
    filtered = [value for value in values if value]
    return separator.join(filtered) if filtered else None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.replace(",", "").strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError:
        try:
            as_float = float(normalized)
        except ValueError:
            return None
        return int(as_float) if as_float.is_integer() else None


def _extract_date_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None

    direct = _clean_text(node.text)
    if direct:
        return direct

    month = _node_text(_find_child(node, "month"))
    day = _node_text(_find_child(node, "day"))
    year = _node_text(_find_child(node, "year"))

    if month and day and year:
        return f"{month} {day}, {year}"
    if month and year:
        return f"{month} {year}"
    if year:
        return year
    return None


def parse_clinical_date(date_text: str | None) -> date | None:
    """Parse common ClinicalTrials.gov date strings."""
    normalized = _clean_text(date_text)
    if not normalized:
        return None
    if normalized.lower() in _EMPTY_DATE_TOKENS:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    return None


def _extract_outcomes(root: ET.Element, tag_name: str) -> str | None:
    items: list[str] = []
    for outcome in _find_children(root, tag_name):
        parts: list[str] = []
        for field_name in ("measure", "description", "time_frame"):
            value = _node_text(_find_child(outcome, field_name))
            if value:
                parts.append(f"{field_name}: {value}")
        if parts:
            items.append("; ".join(parts))
    return _joined_or_none(items, separator=" || ")


def _extract_interventions(root: ET.Element) -> str | None:
    items: list[str] = []
    for intervention in _find_children(root, "intervention"):
        parts: list[str] = []
        intervention_type = _node_text(_find_child(intervention, "intervention_type"))
        intervention_name = _node_text(_find_child(intervention, "intervention_name"))
        description = _node_text(_find_child(intervention, "description"))

        if intervention_type:
            parts.append(intervention_type)
        if intervention_name:
            parts.append(intervention_name)
        if description:
            parts.append(f"description: {description}")

        entry = _joined_or_none(parts)
        if entry:
            items.append(entry)
    return _joined_or_none(items, separator=" || ")


def parse_trial_xml(xml_path: Path) -> dict[str, Any]:
    """
    Parse one ClinicalTrials.gov classic XML file into semantic and date fields.

    Returned dict contains internal date fields prefixed with `_` for duration
    label derivation. Date fields are intentionally separate from semantic text.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if _local_name(root.tag) != "clinical_study":
        raise ValueError(f"Unexpected root tag '{root.tag}' in {xml_path}")

    nct_id_from_filename = xml_path.stem
    xml_nct_id = _node_text(_find_path(root, "id_info/nct_id"))
    nct_id_mismatch = bool(xml_nct_id and xml_nct_id != nct_id_from_filename)

    brief_title = _node_text(_find_child(root, "brief_title")) or _node_text(
        _find_child(root, "official_title")
    )
    summary_text = _node_text(_find_path(root, "brief_summary/textblock")) or _node_text(
        _find_child(root, "brief_summary")
    )
    description_text = _node_text(_find_path(root, "detailed_description/textblock")) or _node_text(
        _find_child(root, "detailed_description")
    )

    enrollment_node = _find_child(root, "enrollment")
    enrollment_value = _node_text(enrollment_node)
    enrollment_count = _parse_optional_int(enrollment_value)
    enrollment_type = _clean_text(enrollment_node.attrib.get("type")) if enrollment_node is not None else None

    conditions = [_node_text(node) or "" for node in _find_children(root, "condition")]
    conditions_text = _joined_or_none([item for item in conditions if item])

    return {
        "nct_id": nct_id_from_filename,
        "xml_nct_id": xml_nct_id,
        "nct_id_mismatch": nct_id_mismatch,
        "brief_title": brief_title,
        "summary_text": summary_text,
        "description_text": description_text,
        "primary_outcomes_text": _extract_outcomes(root, "primary_outcome"),
        "secondary_outcomes_text": _extract_outcomes(root, "secondary_outcome"),
        "enrollment_count": enrollment_count,
        "enrollment_type": enrollment_type,
        "phase": _node_text(_find_child(root, "phase")),
        "overall_status": _node_text(_find_child(root, "overall_status")),
        "conditions_text": conditions_text,
        "interventions_text": _extract_interventions(root),
        "_start_date": parse_clinical_date(_extract_date_text(_find_child(root, "start_date"))),
        "_primary_completion_date": parse_clinical_date(
            _extract_date_text(_find_child(root, "primary_completion_date"))
        ),
        "_completion_date": parse_clinical_date(_extract_date_text(_find_child(root, "completion_date"))),
    }


def derive_duration_label(
    start_date: date | None,
    primary_completion_date: date | None,
    completion_date: date | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compute duration labels in days and months."""
    if start_date is None:
        return None, "missing_start_date"

    if primary_completion_date is not None:
        end_date = primary_completion_date
        duration_source = "primary_completion_date"
    elif completion_date is not None:
        end_date = completion_date
        duration_source = "completion_date"
    else:
        return None, "missing_completion_date"

    duration_days = (end_date - start_date).days
    if duration_days < 0:
        return None, "negative_duration"

    duration_months = round(duration_days / 30.44, 2)
    return {
        "duration_days": duration_days,
        "duration_months": duration_months,
        "duration_source": duration_source,
    }, None
