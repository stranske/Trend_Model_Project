"""Validation helpers for result explanation outputs."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable

from trend_analysis.llm.result_metrics import (
    MetricEntry,
    available_metric_keywords,
    known_metric_keywords,
)

RESULT_DISCLAIMER = (
    "This is analytical output, not financial advice. Always verify metrics independently."
)

_CITATION_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?%?)\s*\[from\s+(?P<source>[^\]]+)\]",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"\[from\s+(?P<source>[^\]]+)\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?%?")
_DATE_RE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})-(?P<month>0?[1-9]|1[0-2])" r"(?:-(?P<day>0?[1-9]|[12]\d|3[01]))?\b"
)
_YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>(?:19|20)\d{2})\s*(?:-|to|through|\u2013|\u2014)\s*(?P<end>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_YEAR_RANGE_SLASH_RE = re.compile(r"\b(?P<start>(?:19|20)\d{2})\s*/\s*(?P<end>(?:19|20)\d{2})\b")
_SHORT_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start>(?:19|20)\d{2})\s*(?:-|/|\u2013|\u2014)\s*(?P<end>\d{2})(?![-/\d])"
)
_SLASH_DATE_RE = re.compile(
    r"\b(?P<month>0?[1-9]|1[0-2])/(?P<day>0?[1-9]|[12]\d|3[01])" r"(?:/(?P<year>(?:19|20)\d{2}))?\b"
)
_DASH_DATE_RE = re.compile(
    r"\b(?P<month>0?[1-9]|1[0-2])-(?P<day>0?[1-9]|[12]\d|3[01])-(?P<year>(?:19|20)\d{2})\b"
)
_MONTH_YEAR_RE = re.compile(r"\b(?P<month>0?[1-9]|1[0-2])\s*[/\-]\s*(?P<year>(?:19|20)\d{2})\b")
_YEAR_SLASH_DATE_RE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})/(?P<month>0?[1-9]|1[0-2])" r"(?:/(?P<day>0?[1-9]|[12]\d|3[01]))?\b"
)
_QUESTION_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_UNAVAILABLE_RE = re.compile(
    r"requested data is unavailable in the analysis output",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResultClaimIssue:
    kind: str
    message: str
    detail: dict[str, object]


def serialize_claim_issue(issue: ResultClaimIssue) -> dict[str, object]:
    """Convert a claim issue into a JSON-friendly dict."""

    detail = issue.detail if isinstance(issue.detail, dict) else {"detail": issue.detail}
    safe_detail = json.loads(json.dumps(detail, default=str))
    return {"kind": issue.kind, "message": issue.message, "detail": safe_detail}


def ensure_result_disclaimer(text: str) -> str:
    """Ensure the required disclaimer is present in the response."""

    if RESULT_DISCLAIMER in text:
        return text
    cleaned = text.rstrip()
    if cleaned:
        return f"{cleaned}\n\n{RESULT_DISCLAIMER}"
    return RESULT_DISCLAIMER


def apply_metric_citations(
    text: str,
    entries: Iterable[MetricEntry],
    *,
    tolerance: float = 1e-4,
) -> str:
    """Append citations for uncited numeric values when matches are available."""

    entries_list = list(entries)
    if not text or not entries_list:
        return text
    citation_spans = [match.span() for match in _CITATION_RE.finditer(text)]
    parts: list[str] = []
    cursor = 0
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        parts.append(text[cursor:start])
        parts.append(text[start:end])
        if not _overlaps((start, end), citation_spans):
            sources = _find_matching_sources(
                match.group(0),
                entries_list,
                tolerance=tolerance,
            )
            if sources:
                parts.append(f" [from {', '.join(sources)}]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def format_discrepancy_log(issues: Iterable[ResultClaimIssue]) -> str:
    """Render discrepancy issues into a user-facing log block."""

    lines = ["Discrepancy log:"]
    issues_list = list(issues)
    for issue in issues_list:
        lines.append(f"- {issue.kind}: {issue.message}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def append_discrepancy_log(text: str, issues: Iterable[ResultClaimIssue]) -> str:
    """Append the discrepancy log to the explanation text."""

    log_block = format_discrepancy_log(issues)
    if not log_block:
        return text
    cleaned = text.rstrip()
    if cleaned:
        return f"{cleaned}\n\n{log_block}"
    return log_block


def postprocess_result_text(
    text: str,
    entries: Iterable[MetricEntry],
    *,
    logger: logging.Logger | None = None,
    tolerance: float = 1e-4,
) -> tuple[str, list[ResultClaimIssue]]:
    """Apply citations, validate claims, and append discrepancy logs."""

    if is_unavailability_response(text):
        output = ensure_result_disclaimer(text)
        return output, []
    cited_text = apply_metric_citations(text, entries, tolerance=tolerance)
    issues = validate_result_claims(cited_text, entries, logger=logger, tolerance=tolerance)
    output = append_discrepancy_log(cited_text, issues)
    output = ensure_result_disclaimer(output)
    return output, issues


def detect_unavailable_metric_requests(
    questions: str,
    entries: Iterable[MetricEntry],
) -> list[str]:
    """Return normalized metric keywords requested but unavailable."""

    normalized_questions = _normalize_question_text(questions)
    available = available_metric_keywords(entries)
    known = known_metric_keywords()
    requested = {kw for kw in known if f" {kw} " in normalized_questions}
    missing = sorted(requested - available)
    return missing


def is_unavailability_response(text: str) -> bool:
    """Return True if the text indicates unavailable data."""

    return bool(_UNAVAILABLE_RE.search(text))


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
    date_spans = _date_like_number_spans(text)

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

    if not citation_spans:
        issues.append(
            ResultClaimIssue(
                kind="missing_citation",
                message="Explanation does not include any cited metric values.",
                detail={},
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
        if _overlaps(match.span(), date_spans):
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
    hallucination_kinds = {
        "unknown_source",
        "value_mismatch",
        "missing_value",
        "uncited_value",
        "missing_citation",
    }
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


def _date_like_number_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _DATE_RE.finditer(text):
        spans.append(match.span("year"))
        if match.group("month"):
            spans.append(match.span("month"))
        if match.group("day"):
            spans.append(match.span("day"))
    for match in _YEAR_RANGE_RE.finditer(text):
        spans.append(match.span("start"))
        spans.append(match.span("end"))
    for match in _YEAR_RANGE_SLASH_RE.finditer(text):
        spans.append(match.span("start"))
        spans.append(match.span("end"))
    for match in _SHORT_YEAR_RANGE_RE.finditer(text):
        spans.append(match.span("start"))
        spans.append(match.span("end"))
    for match in _SLASH_DATE_RE.finditer(text):
        spans.append(match.span("month"))
        spans.append(match.span("day"))
        if match.group("year"):
            spans.append(match.span("year"))
    for match in _DASH_DATE_RE.finditer(text):
        spans.append(match.span("month"))
        spans.append(match.span("day"))
        spans.append(match.span("year"))
    for match in _MONTH_YEAR_RE.finditer(text):
        spans.append(match.span("month"))
        spans.append(match.span("year"))
    for match in _YEAR_SLASH_DATE_RE.finditer(text):
        spans.append(match.span("year"))
        spans.append(match.span("month"))
        if match.group("day"):
            spans.append(match.span("day"))
    return spans


def _find_matching_sources(
    value_text: str,
    entries: Iterable[MetricEntry],
    *,
    tolerance: float,
) -> list[str]:
    is_percent = value_text.endswith("%")
    try:
        value = float(value_text.rstrip("%"))
    except ValueError:
        return []
    sources: list[str] = []
    for entry in entries:
        if isinstance(entry.value, (int, float)):
            expected = float(entry.value)
            expected = expected * 100 if is_percent else expected
            if _close_enough(value, expected, tolerance):
                sources.append(entry.source)
    return sorted(set(sources))


def _normalize_question_text(text: str) -> str:
    normalized = _QUESTION_TOKEN_RE.sub(" ", text.lower()).strip()
    return f" { ' '.join(normalized.split()) } "


__all__ = [
    "RESULT_DISCLAIMER",
    "ResultClaimIssue",
    "append_discrepancy_log",
    "apply_metric_citations",
    "detect_unavailable_metric_requests",
    "ensure_result_disclaimer",
    "format_discrepancy_log",
    "is_unavailability_response",
    "postprocess_result_text",
    "validate_result_claims",
    "detect_result_hallucinations",
]
