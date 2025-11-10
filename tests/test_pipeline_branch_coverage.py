"""Additional branch coverage tests for ``trend_analysis.pipeline``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest

from trend_analysis import pipeline
from trend_analysis.pipeline import RiskStatsConfig
from trend_analysis.risk import RiskDiagnostics


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    data = {
        "Date": dates,
        "Fund_A": [0.01, 0.02, 0.015, 0.017, np.nan, 0.011],
        "Fund_B": [0.005, 0.007, np.nan, 0.006, 0.008, 0.01],
        "RF": [0.0005, 0.0004, 0.0003, 0.0006, 0.0005, 0.0004],
    }
    return pd.DataFrame(data)


def _stub_diagnostics(columns: Iterator[str]) -> RiskDiagnostics:
    cols = list(columns)
    index = pd.date_range("2024-01-31", periods=6, freq="M")
    asset_vol = pd.DataFrame(0.1, index=index, columns=cols)
    portfolio = pd.Series(0.1, index=index, name="portfolio")
    turnover = pd.Series([], dtype=float, name="turnover")
    scale = pd.Series(1.0, index=cols)
    return RiskDiagnostics(asset_volatility=asset_vol, portfolio_volatility=portfolio, turnover=turnover, turnover_value=0.0, scale_factors=scale)


def test_preprocessing_summary_monthly_branch() -> None:
    summary = pipeline._preprocessing_summary("M", normalised=False, missing_summary="none")
    assert "Monthly (month-end)" in summary
    assert "Missing data: none" in summary


def test_resolve_sample_split_returns_existing_keys() -> None:
    df = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=2, freq="M")})
    split_cfg = {"in_start": "2024-01", "in_end": "2024-02", "out_start": "2024-03", "out_end": "2024-04"}
    result = pipeline._resolve_sample_split(df, split_cfg)
    assert result == split_cfg


def test_prepare_input_data_coerces_dates() -> None:
    df = pd.DataFrame({"Date": ["2024-01-31", "2024-02-29"], "Fund": [0.01, 0.02]})
    prepared, summary, missing, normalised = pipeline._prepare_input_data(
        df,
        date_col="Date",
        missing_policy=None,
        missing_limit=None,
    )
    assert pd.api.types.is_datetime64_ns_dtype(prepared["Date"].dtype)
    assert summary.code == "M"
    assert missing.default_policy == "drop"
    assert normalised is False


def test_single_period_run_adds_avg_corr_metric() -> None:
    df = _sample_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"])
    stats_cfg.extra_metrics = ["AvgCorr"]
    result = pipeline.single_period_run(df, "2024-01", "2024-03", stats_cfg=stats_cfg)
    assert "AvgCorr" in result.columns


def test_run_analysis_df_none_returns_none() -> None:
    assert pipeline._run_analysis(
        None,
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        target_vol=0.1,
        monthly_cost=0.0,
    ) is None


def test_run_analysis_requires_date_column() -> None:
    df = pd.DataFrame({"Fund": [0.1, 0.2]})
    with pytest.raises(ValueError):
        pipeline._run_analysis(
            df,
            "2024-01",
            "2024-02",
            "2024-03",
            "2024-04",
            target_vol=0.1,
            monthly_cost=0.0,
        )


def test_run_analysis_returns_none_when_windows_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_frame()

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Any, Any, bool]:
        prepared = df.copy()
        summary = SimpleNamespace(code="M", label="Monthly", target="M", target_label="Monthly", resampled=False)
        missing = SimpleNamespace(
            default_policy="drop",
            policy={},
            default_limit=None,
            limit={},
            dropped_assets=set(),
            filled_cells={},
            total_filled=0,
            summary="none",
        )
        return prepared, summary, missing, False

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)
    result = pipeline._run_analysis(
        df,
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        target_vol=0.1,
        monthly_cost=0.0,
    )
    assert result is None


def test_run_analysis_na_policy_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_frame()

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Any, Any, bool]:
        prepared = df.copy()
        summary = SimpleNamespace(code="M", label="Monthly", target="M", target_label="Monthly", resampled=False)
        missing = SimpleNamespace(
            default_policy="drop",
            policy={},
            default_limit=None,
            limit={},
            dropped_assets=set(),
            filled_cells={},
            total_filled=0,
            summary="none",
        )
        return prepared, summary, missing, False

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)

    def fake_single_period_run(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"Sharpe": [1.0, 0.5]}, index=["Fund_A", "Fund_B"])

    def fake_compute_trend_signals(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=df.set_index("Date").index, columns=["Fund_A", "Fund_B"])

    def fake_compute_constrained_weights(*args: Any, **kwargs: Any) -> tuple[pd.Series, RiskDiagnostics]:
        weights = pd.Series({"Fund_A": 0.5, "Fund_B": 0.5}, dtype=float)
        return weights, _stub_diagnostics(weights.index)

    monkeypatch.setattr(pipeline, "single_period_run", fake_single_period_run)
    monkeypatch.setattr(pipeline, "compute_trend_signals", fake_compute_trend_signals)
    monkeypatch.setattr(pipeline, "compute_constrained_weights", fake_compute_constrained_weights)
    monkeypatch.setattr(pipeline, "realised_volatility", lambda *a, **k: pd.DataFrame({"portfolio": [0.1]}, index=pd.RangeIndex(1)))
    monkeypatch.setattr(pipeline, "build_regime_payload", lambda **_: {})
    monkeypatch.setattr(pipeline, "information_ratio", lambda *_, **__: 0.1)

    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe", "AvgCorr"])
    stats_cfg.extra_metrics = ["AvgCorr"]
    stats_cfg.na_as_zero_cfg = {
        "enabled": True,
        "max_missing_per_window": 2,
        "max_consecutive_gap": 1,
    }

    result = pipeline._run_analysis(
        df,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        target_vol=0.1,
        monthly_cost=0.0,
        warmup_periods=1,
        stats_cfg=stats_cfg,
    )

    assert result is not None
    assert "risk_diagnostics" in result
    diag = result["risk_diagnostics"]
    assert isinstance(diag, dict)
    assert "scale_factors" in diag


def test_run_analysis_information_ratio_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_frame().fillna(0.0).assign(Fund_C=lambda frame: frame["Fund_A"] * 0.8)

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Any, Any, bool]:
        prepared = df.copy()
        summary = SimpleNamespace(code="M", label="Monthly", target="M", target_label="Monthly", resampled=False)
        missing = SimpleNamespace(
            default_policy="drop",
            policy={},
            default_limit=None,
            limit={},
            dropped_assets=set(),
            filled_cells={},
            total_filled=0,
            summary="none",
        )
        return prepared, summary, missing, False

    def fake_compute_constrained_weights(*args: Any, **kwargs: Any) -> tuple[pd.Series, RiskDiagnostics]:
        weights = pd.Series({"Fund_A": 0.4, "Fund_B": 0.4, "Fund_C": 0.2}, dtype=float)
        return weights, _stub_diagnostics(weights.index)

    def raising_information_ratio(first: Any, second: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(first, pd.Series):
            return 0.0
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)
    monkeypatch.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [1.0, 0.5, 0.3]}, index=["Fund_A", "Fund_B", "Fund_C"]))
    monkeypatch.setattr(pipeline, "compute_trend_signals", lambda *a, **k: pd.DataFrame(0.0, index=df.set_index("Date").index, columns=["Fund_A", "Fund_B", "Fund_C"]))
    monkeypatch.setattr(pipeline, "compute_constrained_weights", fake_compute_constrained_weights)
    monkeypatch.setattr(pipeline, "realised_volatility", lambda *a, **k: pd.DataFrame({"portfolio": [0.1]}, index=pd.RangeIndex(1)))
    monkeypatch.setattr(pipeline, "build_regime_payload", lambda **_: {})
    monkeypatch.setattr(pipeline, "information_ratio", raising_information_ratio)

    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe", "AvgCorr"])
    stats_cfg.extra_metrics = ["AvgCorr"]

    result = pipeline._run_analysis(
        df,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        target_vol=0.1,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
    )

    assert result is not None
    # Benchmark IR fallback should still be present even when information_ratio fails.
    assert "benchmark_ir" in result

