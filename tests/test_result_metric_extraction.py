"""Tests for analysis result metric extraction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trend_analysis.llm.result_metrics import (
    MetricEntry,
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


def test_extract_metric_catalog_includes_risk_diagnostics_scalars() -> None:
    result = {
        "risk_diagnostics": {
            "turnover": 0.12,
            "turnover_value": 0.2,
            "transaction_cost": 0.01,
            "per_trade_bps": 5.0,
        }
    }

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "risk_diagnostics.turnover" in paths
    assert "risk_diagnostics.turnover_value" in paths
    assert "risk_diagnostics.transaction_cost" in paths
    assert "risk_diagnostics.per_trade_bps" in paths
    assert "[from risk_diagnostics]" in format_metric_catalog(entries)


def test_extract_metric_catalog_reads_risk_diagnostics_attributes() -> None:
    risk_diag = SimpleNamespace(turnover=0.08, cost=0.004, half_spread_bps=1.5)
    result = {"risk_diagnostics": risk_diag}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "risk_diagnostics.turnover" in paths
    assert "risk_diagnostics.cost" in paths
    assert "risk_diagnostics.half_spread_bps" in paths


def test_extract_metric_catalog_reads_risk_diagnostics_series() -> None:
    risk_diag = pd.Series({"turnover": 0.07, "transaction_costs": 0.02})
    result = {"risk_diagnostics": risk_diag}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "risk_diagnostics.turnover" in paths
    assert "risk_diagnostics.transaction_costs" in paths


def test_extract_metric_catalog_reads_risk_diagnostics_from_details() -> None:
    result = {"details": {"risk_diagnostics": {"turnover": 0.2, "cost": 0.01}}}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}

    assert "risk_diagnostics.turnover" in paths
    assert "risk_diagnostics.cost" in paths


def test_extract_metric_catalog_adds_turnover_series_summaries() -> None:
    result = {"turnover": pd.Series([0.05, 0.1, 0.2])}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}
    catalog = format_metric_catalog(entries)

    assert "turnover.latest" in paths
    assert "turnover.mean" in paths
    assert "[from turnover_series]" in catalog


def test_extract_metric_catalog_adds_turnover_series_from_details() -> None:
    result = {"details": {"turnover": pd.Series([0.02, 0.08])}}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}
    catalog = format_metric_catalog(entries)

    assert "turnover.latest" in paths
    assert "turnover.mean" in paths
    assert "[from turnover_series]" in catalog


def test_extract_metric_catalog_adds_turnover_series_from_risk_diagnostics() -> None:
    result = {"risk_diagnostics": {"turnover": pd.Series([0.03, 0.07, 0.12])}}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}
    catalog = format_metric_catalog(entries)

    assert "turnover.latest" in paths
    assert "turnover.mean" in paths
    assert "[from turnover_series]" in catalog


def test_extract_metric_catalog_adds_turnover_scalar_from_details() -> None:
    result = {"details": {"turnover": 0.12}}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}
    catalog = format_metric_catalog(entries)

    assert "turnover.value" in paths
    assert "[from turnover_scalar]" in catalog


def test_extract_metric_catalog_adds_turnover_scalar_from_risk_diagnostics() -> None:
    result = {"risk_diagnostics": {"turnover_value": 0.18}}

    entries = extract_metric_catalog(result)
    paths = {entry.path for entry in entries}
    catalog = format_metric_catalog(entries)

    assert "risk_diagnostics.turnover_value" in paths
    assert "turnover.value" in paths
    assert "[from turnover_scalar]" in catalog


def test_format_metric_catalog_defaults_missing_source() -> None:
    entries = [
        MetricEntry(path="risk_diagnostics.turnover", value=0.2, source=""),
    ]

    catalog = format_metric_catalog(entries)

    assert "risk_diagnostics.turnover" in catalog
    assert "[from unknown]" in catalog
