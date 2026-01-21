"""Tests for compact metric catalog selection."""

from __future__ import annotations

from trend_analysis.llm.result_metrics import (
    compact_metric_catalog,
    extract_metric_catalog,
)


def _make_result(num_funds: int = 8) -> dict:
    weights = {
        f"Fund{idx}": weight
        for idx, weight in enumerate(
            [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01][:num_funds]
        )
    }
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
