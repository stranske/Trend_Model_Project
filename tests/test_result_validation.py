"""Tests for result explanation validation."""

from __future__ import annotations

from trend_analysis.llm.result_metrics import MetricEntry
from trend_analysis.llm.result_validation import (
    detect_result_hallucinations,
    validate_result_claims,
)


def test_validate_result_claims_accepts_matching_citation() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "CAGR was 8% [from out_sample_stats]."

    assert validate_result_claims(text, entries) == []


def test_detect_result_hallucinations_flags_missing_metric() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "CAGR was 12% [from out_sample_stats]."

    issues = detect_result_hallucinations(text, entries)
    kinds = {issue.kind for issue in issues}

    assert "value_mismatch" in kinds


def test_validate_result_claims_flags_uncited_values() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "CAGR was 8%."

    issues = validate_result_claims(text, entries)

    assert any(issue.kind == "uncited_value" for issue in issues)


def test_validate_result_claims_flags_missing_citations() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "Performance was strong over the period."

    issues = validate_result_claims(text, entries)

    assert any(issue.kind == "missing_citation" for issue in issues)
