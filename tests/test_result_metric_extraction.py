"""Tests for analysis result metric extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

from trend_analysis.llm.result_metrics import (
    extract_metric_catalog,
    format_metric_catalog,
)


def test_extract_metric_catalog_collects_stats_weights_benchmarks() -> None:
    result = {
        "out_sample_stats": {
            "FundA": SimpleNamespace(
                cagr=0.12,
                vol=0.2,
                sharpe=1.1,
                sortino=1.4,
                information_ratio=0.3,
                max_drawdown=-0.25,
                is_avg_corr=None,
                os_avg_corr=0.1,
            ),
        },
        "in_sample_stats": {
            "FundA": {
                "cagr": 0.08,
                "vol": 0.18,
                "sharpe": 0.9,
                "sortino": 1.2,
                "information_ratio": 0.2,
                "max_drawdown": -0.2,
            },
        },
        "out_ew_stats": (0.1, 0.15, 1.0, 1.1, 0.25, -0.3),
        "fund_weights": {"FundA": 0.6, "FundB": 0.4},
        "benchmark_ir": {"SPX": {"FundA": 0.05, "equal_weight": 0.12}},
    }

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "out_sample_stats.FundA.cagr" in paths
    assert "in_sample_stats.FundA.max_drawdown" in paths
    assert "out_ew_stats.cagr" in paths
    assert "fund_weights.FundA" in paths
    assert "benchmark_ir.SPX.FundA" in paths


def test_extract_metric_catalog_skips_none_and_nan() -> None:
    result = {
        "out_sample_stats": {
            "FundA": SimpleNamespace(
                cagr=None,
                vol=float("nan"),
                sharpe=1.2,
                sortino=1.3,
                information_ratio=0.1,
                max_drawdown=-0.1,
            )
        }
    }

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "out_sample_stats.FundA.cagr" not in paths
    assert "out_sample_stats.FundA.vol" not in paths
    assert "out_sample_stats.FundA.sharpe" in paths


def test_format_metric_catalog_includes_sources() -> None:
    result = {
        "out_sample_stats": {
            "FundA": {
                "cagr": 0.1,
                "vol": 0.2,
                "sharpe": 1.0,
                "sortino": 1.1,
                "information_ratio": 0.3,
                "max_drawdown": -0.2,
            }
        }
    }

    entries = extract_metric_catalog(result)
    catalog = format_metric_catalog(entries)

    assert "out_sample_stats.FundA.cagr" in catalog
    assert "[from out_sample_stats]" in catalog
