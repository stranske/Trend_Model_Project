import numpy as np
import pandas as pd

import trend_analysis.core.rank_selection as rs


def _sample_window() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    data = {
        "Date": dates,
        "FundA": np.linspace(0.01, 0.06, len(dates)),
        "FundB": np.linspace(0.015, 0.05, len(dates)),
        "FundC": np.linspace(0.02, 0.07, len(dates)),
    }
    df = pd.DataFrame(data)
    return df.set_index("Date")[["FundA", "FundB", "FundC"]]


def test_rank_selector_reuses_metric_bundle() -> None:
    rs.reset_selector_cache()
    window = _sample_window()
    stats_cfg = rs.RiskStatsConfig()
    window_key = rs.make_window_key("2020-01", "2020-06", window.columns, stats_cfg)

    rs.rank_select_funds(
        window,
        stats_cfg,
        inclusion_approach="top_n",
        n=2,
        score_by="Sharpe",
        window_key=window_key,
    )
    assert rs.selector_cache_hits == 0

    rs.rank_select_funds(
        window,
        stats_cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="Sharpe",
        window_key=window_key,
    )
    assert rs.selector_cache_hits >= 1

    bundle = rs.get_window_metric_bundle(window_key)
    assert bundle is not None
    frame = bundle.as_frame()
    assert "Sharpe" in frame.columns

    rs.rank_select_funds(
        window,
        stats_cfg,
        inclusion_approach="top_n",
        n=2,
        score_by="AvgCorr",
        window_key=window_key,
        bundle=bundle,
    )
    assert bundle.cov_payload is not None

    hits_before = rs.selector_cache_hits
    rs.rank_select_funds(
        window,
        stats_cfg,
        inclusion_approach="top_n",
        n=2,
        score_by="AvgCorr",
        window_key=window_key,
        bundle=bundle,
    )
    assert rs.selector_cache_hits > hits_before
