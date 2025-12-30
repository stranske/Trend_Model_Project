import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig, canonical_metric_list
from trend_analysis.pipeline import _run_analysis

RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def make_df():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,  # proxy risk-free for stats
            "A": 0.01,
            "B": 0.02,
        }
    )
    # introduce a single NaN in A and B in different windows
    df.loc[df.index[2], "A"] = np.nan  # within IS
    df.loc[df.index[9], "B"] = np.nan  # within OOS
    return df


def test_strict_mode_drops_funds_with_nans():
    df = make_df()
    res = _run_analysis(
        df,
        "2020-01",
        "2020-06",
        "2020-07",
        "2020-12",
        1.0,
        0.0,
        **RUN_KWARGS,
    )
    # A has NaN in IS, B has NaN in OOS => both dropped => None
    assert res is None


def test_na_as_zero_retains_and_fills():
    df = make_df()
    # build stats_cfg carrying na_as_zero policy
    stats_cfg = RiskStatsConfig(
        metrics_to_run=canonical_metric_list(["annual_return", "volatility"]),
        risk_free=0.0,
    )
    stats_cfg.na_as_zero_cfg = {
        "enabled": True,
        "max_missing_per_window": 2,
        "max_consecutive_gap": 1,
    }
    res = _run_analysis(
        df,
        "2020-01",
        "2020-06",
        "2020-07",
        "2020-12",
        1.0,
        0.0,
        stats_cfg=stats_cfg,
        **RUN_KWARGS,
    )
    assert res is not None
    funds = res["selected_funds"]
    # Both A and B should be retained under the tolerance
    assert set(funds) == {"A", "B"}


def test_na_as_zero_rejects_large_consecutive_gap():
    df = make_df()
    # Create a consecutive gap exceeding the configured tolerance while keeping
    # the total number of missing points within the allowed budget.
    df.loc[df.index[1:3], "A"] = np.nan  # two-period gap in-sample

    stats_cfg = RiskStatsConfig(
        metrics_to_run=canonical_metric_list(["annual_return", "volatility"]),
        risk_free=0.0,
    )
    stats_cfg.na_as_zero_cfg = {
        "enabled": True,
        "max_missing_per_window": 2,
        "max_consecutive_gap": 1,
    }

    res = _run_analysis(
        df,
        "2020-01",
        "2020-06",
        "2020-07",
        "2020-12",
        1.0,
        0.0,
        stats_cfg=stats_cfg,
        **RUN_KWARGS,
    )

    assert res is not None
    assert res["selected_funds"] == ["B"]
