"""Validation helpers for result explanation outputs."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable

from trend_analysis.llm.result_metrics import MetricEntry

RESULT_DISCLAIMER = (
    "This is analytical output, not financial advice. Always verify metrics independently."
)

_CITATION_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?%?)\s*\[from\s+(?P<source>[^\]]+)\]",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"\[from\s+(?P<source>[^\]]+)\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?%?")


@dataclass(frozen=True)
class ResultClaimIssue:
    kind: str
    message: str
    detail: dict[str, object]


def ensure_result_disclaimer(text: str) -> str:
    """Ensure the required disclaimer is present in the response."""

    if RESULT_DISCLAIMER in text:
        return text
    cleaned = text.rstrip()
    if cleaned:
        return f"{cleaned}\n\n{RESULT_DISCLAIMER}"
    return RESULT_DISCLAIMER


def validate_result_claims(
    text: str,
    entries: Iterable[MetricEntry],
    *,
    logger: logging.Logger | None = None,
    tolerance: float = 1e-4,
) -> list[ResultClaimIssue]:
    """Validate cited values against the metric catalog and log discrepancies."""

    issues: list[ResultClaimIssue] = []
    source_map: dict[str, list[MetricEntry]] = {}
    for entry in entries:
        source_map.setdefault(entry.source, []).append(entry)

    citation_spans: list[tuple[int, int]] = []
    value_spans: list[tuple[int, int]] = []

    for match in _CITATION_RE.finditer(text):
        source = match.group("source").strip()
        value_text = match.group("value").strip()
        citation_spans.append(match.span())
        value_spans.append(match.span("value"))
        if source not in source_map:
            issues.append(
                ResultClaimIssue(
                    kind="unknown_source",
                    message=f"Unknown citation source '{source}'.",
                    detail={"source": source, "value": value_text},
                )
            )
            continue
        if not _match_cited_value(value_text, source_map[source], tolerance=tolerance):
            issues.append(
                ResultClaimIssue(
                    kind="value_mismatch",
                    message=f"Cited value '{value_text}' not found in source '{source}'.",
                    detail={"source": source, "value": value_text},
                )
            )

    for match in _SOURCE_RE.finditer(text):
        if _overlaps(match.span(), citation_spans):
            continue
        source = match.group("source").strip()
        issues.append(
            ResultClaimIssue(
                kind="missing_value",
                message=f"Citation source '{source}' missing a value.",
                detail={"source": source},
            )
        )

    for match in _NUMBER_RE.finditer(text):
        if _overlaps(match.span(), value_spans):
            continue
        value_text = match.group(0)
        issues.append(
            ResultClaimIssue(
                kind="uncited_value",
                message=f"Value '{value_text}' lacks a citation.",
                detail={"value": value_text},
            )
        )

    if logger is not None:
        for issue in issues:
            logger.warning("Result claim issue (%s): %s", issue.kind, issue.message)

    return issues


def detect_result_hallucinations(
    text: str,
    entries: Iterable[MetricEntry],
    *,
    logger: logging.Logger | None = None,
) -> list[ResultClaimIssue]:
    """Return potential hallucinations in a result explanation."""

    issues = validate_result_claims(text, entries, logger=None)
    hallucination_kinds = {"unknown_source", "value_mismatch", "missing_value", "uncited_value"}
    hallucinations = [issue for issue in issues if issue.kind in hallucination_kinds]
    if logger is not None:
        for issue in hallucinations:
            logger.warning("Potential hallucination (%s): %s", issue.kind, issue.message)
    return hallucinations


def _match_cited_value(
    value_text: str,
    entries: Iterable[MetricEntry],
    *,
    tolerance: float,
) -> bool:
    is_percent = value_text.endswith("%")
    try:
        value = float(value_text.rstrip("%"))
    except ValueError:
        return False
    for entry in entries:
        if isinstance(entry.value, (int, float)):
            expected = float(entry.value)
            expected = expected * 100 if is_percent else expected
            if _close_enough(value, expected, tolerance):
                return True
    return False


def _close_enough(value: float, expected: float, tolerance: float) -> bool:
    return math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance)


def _overlaps(span: tuple[int, int], spans: Iterable[tuple[int, int]]) -> bool:
    start, end = span
    for other_start, other_end in spans:
        if start < other_end and other_start < end:
            return True
    return False


__all__ = [
    "RESULT_DISCLAIMER",
    "ResultClaimIssue",
    "ensure_result_disclaimer",
    "validate_result_claims",
    "detect_result_hallucinations",
]
