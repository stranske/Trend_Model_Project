"""Tests for compact metric catalog selection."""

from __future__ import annotations

from trend_analysis.llm.result_metrics import (
    MetricEntry,
    compact_metric_catalog,
    extract_metric_catalog,
    format_metric_catalog,
)


def _make_result(num_funds: int = 8) -> dict:
    weights = {f"Fund{idx}": 1.0 / (idx + 1) for idx in range(num_funds)}
    stats = {
        fund: {
            "cagr": 0.05 + idx * 0.01,
            "vol": 0.15 + idx * 0.01,
            "sharpe": 1.0 + idx * 0.1,
            "sortino": 1.2 + idx * 0.1,
            "information_ratio": 0.2 + idx * 0.01,
            "max_drawdown": -0.2 - idx * 0.01,
        }
        for idx, fund in enumerate(weights.keys())
    }
    return {
        "out_sample_stats": stats,
        "out_ew_stats": (0.1, 0.2, 1.0, 1.1, 0.3, -0.2),
        "fund_weights": weights,
        "benchmark_ir": {"SPX": {"Fund0": 0.1}},
        "risk_diagnostics": {"turnover": 0.15},
    }


def test_compact_metric_catalog_keeps_top_weights_and_stats() -> None:
    entries = extract_metric_catalog(_make_result())
    compacted = compact_metric_catalog(
        entries,
        max_funds=2,
        max_weights=2,
        max_entries=25,
    )
    paths = {entry.path for entry in compacted}

    assert "out_ew_stats.cagr" in paths
    assert "benchmark_ir.SPX.Fund0" in paths
    assert "risk_diagnostics.turnover" in paths
    assert "fund_weights.Fund0" in paths
    assert "fund_weights.Fund1" in paths
    assert "out_sample_stats.Fund0.cagr" in paths
    assert "out_sample_stats.Fund1.cagr" in paths
    assert "out_sample_stats.Fund7.cagr" not in paths
    assert len(compacted) <= 25


def test_compact_metric_catalog_includes_question_fund() -> None:
    entries = extract_metric_catalog(_make_result())
    compacted = compact_metric_catalog(
        entries,
        questions="Focus on Fund4 performance.",
        max_funds=2,
        max_weights=1,
        max_entries=30,
    )
    paths = {entry.path for entry in compacted}

    assert "out_sample_stats.Fund4.cagr" in paths
    assert "out_sample_stats.Fund0.cagr" in paths


def test_compact_metric_catalog_keeps_portfolio_stats_and_top_weights() -> None:
    result = _make_result(num_funds=12)
    result["out_user_stats"] = (0.02, 0.1, 0.8, 0.9, 0.1, -0.05)
    entries = extract_metric_catalog(result)
    compacted = compact_metric_catalog(
        entries,
        max_funds=3,
        max_weights=2,
        max_entries=20,
    )
    paths = {entry.path for entry in compacted}

    assert "out_user_stats.cagr" in paths
    assert "fund_weights.Fund0" in paths
    assert "fund_weights.Fund1" in paths


def test_compact_metric_catalog_prioritizes_weights_over_misc_entries() -> None:
    entries = extract_metric_catalog(_make_result())
    compacted = compact_metric_catalog(
        entries,
        max_funds=2,
        max_weights=1,
        max_entries=8,
    )
    paths = {entry.path for entry in compacted}

    assert "out_ew_stats.cagr" in paths
    assert "benchmark_ir.SPX.Fund0" in paths
    assert "fund_weights.Fund0" in paths
    assert "risk_diagnostics.turnover" not in paths


def test_compact_metric_catalog_bounds_formatted_lines() -> None:
    entries = extract_metric_catalog(_make_result(num_funds=30))
    compacted = compact_metric_catalog(
        entries,
        max_funds=5,
        max_weights=3,
        max_entries=20,
    )
    metric_catalog = format_metric_catalog(compacted)
    lines = [line for line in metric_catalog.splitlines() if line.strip()]

    assert len(lines) <= 20
    assert "fund_weights.Fund0" in metric_catalog
    assert "fund_weights.Fund1" in metric_catalog


def test_format_metric_catalog_sanitizes_multiline_values() -> None:
    entries = [
        MetricEntry(path="notes.detail", value="line1\nline2", source="details"),
        MetricEntry(path="out_ew_stats.cagr", value=0.05, source="out_ew_stats"),
    ]

    metric_catalog = format_metric_catalog(entries)
    lines = [line for line in metric_catalog.splitlines() if line.strip()]

    assert len(lines) == 2
    assert "line1 line2" in metric_catalog
