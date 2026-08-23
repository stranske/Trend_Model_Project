"""Regression gate for benchmark-adjusted alpha metric registration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core.rank_selection import RiskStatsConfig, rank_select_funds
from trend_analysis.core.rank_selection import (
    RiskStatsConfig as PipelineRiskStatsConfig,
)
from trend_analysis.diagnostics import PipelineResult
from trend_analysis.metrics import METRIC_REGISTRY, alpha, annual_return
from trend_analysis.stages import preprocessing as preprocessing_stage
from trend_analysis.stages import selection as selection_stage


def test_alpha_ranks_below_raw_return_for_pure_beta_manager() -> None:
    assert "alpha" in METRIC_REGISTRY

    rng = np.random.default_rng(42)
    n = 36
    benchmark = pd.Series(rng.normal(0.008, 0.004, n), name="SPX")

    manager_a = 2.0 * benchmark
    manager_b = benchmark + 0.0012

    returns = pd.DataFrame(
        {
            "pure_beta": manager_a,
            "positive_alpha": manager_b,
        }
    )

    annual_scores = annual_return(returns)
    alpha_scores = alpha(returns, benchmark=benchmark)

    assert annual_scores["pure_beta"] > annual_scores["positive_alpha"]
    assert alpha_scores["pure_beta"] < alpha_scores["positive_alpha"]
    assert alpha_scores["positive_alpha"] > 0.0


def test_alpha_without_benchmark_returns_nan() -> None:
    returns = pd.DataFrame({"manager": [0.01, 0.02, -0.01, 0.015, 0.005]})
    scores = alpha(returns)
    assert np.isnan(scores["manager"]).item()


def test_alpha_with_scalar_benchmark_returns_nan() -> None:
    returns = pd.DataFrame({"manager": [0.01, 0.02, -0.01, 0.015, 0.005]})
    scores = alpha(returns, benchmark=0.0)
    assert np.isnan(scores["manager"]).item()


def test_rank_selection_forwards_configured_benchmark_to_alpha() -> None:
    rng = np.random.default_rng(7)
    benchmark = pd.Series(rng.normal(0.008, 0.004, 36), name="SPX")
    returns = pd.DataFrame(
        {
            "pure_beta": 2.0 * benchmark,
            "positive_alpha": benchmark + 0.0012,
        }
    )

    selected = rank_select_funds(
        returns,
        RiskStatsConfig(benchmark=benchmark),
        score_by="alpha",
        n=1,
    )

    assert selected == ["positive_alpha"]


def test_select_universe_excludes_configured_benchmark_without_indices_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(11)
    n = 12
    benchmark = pd.Series(rng.normal(0.008, 0.004, n), name="SPX")
    dates = pd.date_range("2020-01-31", periods=n, freq="ME", tz="UTC")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "SPX": benchmark.to_numpy(),
            "manager": (benchmark + 0.0012).to_numpy(),
            "rf": [0.001] * n,
        }
    )
    frame.attrs["calendar_settings"] = {"timezone": None}

    preprocess = preprocessing_stage._prepare_preprocess_stage(
        frame,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=PipelineRiskStatsConfig(risk_free=0.0),
        periods_per_year_override=12,
        allow_risk_free_fallback=True,
    )
    assert not isinstance(preprocess, PipelineResult)

    window = preprocessing_stage._build_sample_windows(
        preprocess,
        in_start="2020-01-31",
        in_end="2020-06-30",
        out_start="2020-07-31",
        out_end="2020-12-31",
    )
    assert not isinstance(window, PipelineResult)

    monkeypatch.setattr(
        selection_stage,
        "single_period_run",
        lambda *args, **kwargs: pd.DataFrame({"Sharpe": [1.0]}, index=["manager"]),
    )

    selection = selection_stage._select_universe(
        preprocess,
        window,
        in_label="2020-01-31",
        in_end_label="2020-06-30",
        selection_mode="all",
        random_n=1,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        benchmarks={"bench": "SPX"},
        seed=1,
        stats_cfg=PipelineRiskStatsConfig(risk_free=0.0),
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )

    assert not isinstance(selection, PipelineResult)
    assert selection.fund_cols == ["manager"]
