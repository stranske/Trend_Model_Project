"""Tests for result explanation validation."""

from __future__ import annotations

from trend_analysis.llm.result_metrics import MetricEntry
from trend_analysis.llm.result_validation import (
    apply_metric_citations,
    detect_result_hallucinations,
    detect_unavailable_metric_requests,
    postprocess_result_text,
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


def test_apply_metric_citations_adds_source_reference() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "CAGR was 8%."

    result = apply_metric_citations(text, entries)

    assert "8% [from out_sample_stats]" in result


def test_postprocess_result_text_appends_discrepancy_log() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]
    text = "CAGR was 12%."

    output, issues = postprocess_result_text(text, entries)

    assert issues
    assert "Discrepancy log:" in output


def test_detect_unavailable_metric_requests_returns_missing_metrics() -> None:
    entries = [
        MetricEntry(
            path="out_sample_stats.portfolio.cagr",
            value=0.08,
            source="out_sample_stats",
        )
    ]

    missing = detect_unavailable_metric_requests("Report alpha and beta.", entries)

    assert "alpha" in missing
    assert "beta" in missing
