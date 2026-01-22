"""Tests for deterministic diagnostics feedback."""

from __future__ import annotations

import re

import pytest

from trend_analysis.llm import build_deterministic_feedback, extract_metric_catalog


def _sample_details() -> dict:
    return {
        "risk_diagnostics": {"turnover_value": 0.92},
        "fund_weights": {"FundA": 0.35, "FundB": 0.12},
        "out_user_stats": {"max_drawdown": -0.35},
        "period_count": 2,
        "out_sample_scaled": None,
    }


def test_build_deterministic_feedback_is_deterministic() -> None:
    details = _sample_details()
    entries = extract_metric_catalog(details)

    first = build_deterministic_feedback(details, entries)
    second = build_deterministic_feedback(details, entries)

    assert first == second


def test_build_deterministic_feedback_flags_and_citations() -> None:
    details = _sample_details()
    entries = extract_metric_catalog(details)

    feedback = build_deterministic_feedback(details, entries)

    assert feedback.startswith("Deterministic diagnostics:")
    assert "turnover" in feedback.lower()
    assert "concentration" in feedback.lower()
    assert "drawdown" in feedback.lower()
    assert "missing" in feedback.lower()

    bullet_lines = [line for line in feedback.splitlines() if line.strip().startswith("- ")]
    assert bullet_lines
    for line in bullet_lines:
        assert "[from " in line
        assert re.search(r"[-+]?\d+(?:\.\d+)?", line)

    banned = ("will ", "should ", "expect", "forecast", "project", "may ", "might ", "could ")
    lower = feedback.lower()
    assert not any(term in lower for term in banned)


def test_build_deterministic_feedback_respects_env_thresholds(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    details = {
        "risk_diagnostics": {"turnover_value": 0.4},
        "fund_weights": {"FundA": 0.15},
        "out_user_stats": {"max_drawdown": -0.1},
    }
    entries = extract_metric_catalog(details)

    monkeypatch.setenv("TREND_EXPLAIN_TURNOVER_WARN", "0.3")
    monkeypatch.setenv("TREND_EXPLAIN_MAX_WEIGHT_WARN", "0.1")
    monkeypatch.setenv("TREND_EXPLAIN_MAX_DD_WARN", "-0.05")

    feedback = build_deterministic_feedback(details, entries)
    lower = feedback.lower()

    assert "high turnover" in lower
    assert "high concentration" in lower
    assert "large drawdown" in lower
