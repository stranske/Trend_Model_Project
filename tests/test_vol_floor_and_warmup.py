from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig, canonical_metric_list
from trend_analysis.pipeline import _run_analysis

RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def _constant_df(include_nan: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": 0.001,
            "B": 0.002,
        }
    )
    if include_nan:
        # Introduce a NaN in the out-of-sample window for A.
        df.loc[df.index[4], "A"] = np.nan
    return df


def test_floor_vol_limits_scaling() -> None:
    df = _constant_df()
    res = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        floor_vol=0.05,
        **RUN_KWARGS,
    )
    assert res is not None
    in_scaled = res["in_sample_scaled"]
    expected = np.full(3, 0.001 * (0.10 / 0.05))
    np.testing.assert_allclose(in_scaled["A"].iloc[:3].to_numpy(), expected)


def test_warmup_zeroes_initial_rows() -> None:
    df = _constant_df()
    res = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        warmup_periods=2,
        **RUN_KWARGS,
    )
    assert res is not None
    out_scaled = res["out_sample_scaled"].iloc[:2]
    assert np.allclose(out_scaled.to_numpy(), 0.0)


def test_nan_returns_become_zero_weight() -> None:
    df = _constant_df(include_nan=True)
    stats_cfg = RiskStatsConfig(
        metrics_to_run=canonical_metric_list(["annual_return", "volatility"]),
        risk_free=0.0,
    )
    setattr(
        stats_cfg,
        "na_as_zero_cfg",
        {"enabled": True, "max_missing_per_window": 2, "max_consecutive_gap": 2},
    )
    res = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        **RUN_KWARGS,
    )
    assert res is not None
    out_scaled = res["out_sample_scaled"]
    # Row with injected NaN should be zero after scaling/fill.
    assert np.allclose(out_scaled.loc[out_scaled.index[1], "A"], 0.0)


def test_negative_floor_and_warmup_inputs_are_clamped() -> None:
    """Negative floor-vol or warmup inputs should behave like zero."""

    df = _constant_df()

    baseline = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        **RUN_KWARGS,
    )
    assert baseline is not None

    clamped = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        floor_vol=-0.25,  # should clamp to zero instead of affecting scaling
        warmup_periods=-5,
        **RUN_KWARGS,
    )
    assert clamped is not None

    np.testing.assert_allclose(
        clamped["in_sample_scaled"].to_numpy(),
        baseline["in_sample_scaled"].to_numpy(),
    )
    np.testing.assert_allclose(
        clamped["out_sample_scaled"].to_numpy(),
        baseline["out_sample_scaled"].to_numpy(),
    )
