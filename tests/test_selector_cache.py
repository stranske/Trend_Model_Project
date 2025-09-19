import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import (RiskStatsConfig,
                                                clear_window_metric_cache,
                                                get_window_metric_bundle,
                                                make_window_key,
                                                rank_select_funds,
                                                selector_cache_stats)


def test_rank_selector_reuses_cached_window_metrics() -> None:
    clear_window_metric_cache()
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    data = pd.DataFrame(
        {
            " Fund A": np.linspace(0.01, 0.12, len(idx)),
            "Fund B": np.linspace(0.02, 0.13, len(idx)),
            "Fund C": np.linspace(0.03, 0.14, len(idx)),
        },
        index=idx,
    )
    cfg = RiskStatsConfig()
    window_key = make_window_key("2020-01", "2020-12", data.columns, cfg)

    rank_select_funds(
        data,
        cfg,
        inclusion_approach="top_n",
        n=2,
        score_by="Sharpe",
        window_key=window_key,
    )

    stats_after_first = selector_cache_stats()
    assert stats_after_first["selector_cache_misses"] >= 1
    assert stats_after_first["selector_cache_hits"] == 0

    bundle = get_window_metric_bundle(window_key)
    assert bundle is not None
    sharpe_frame = bundle.metrics_frame()
    assert "Sharpe" in sharpe_frame.columns

    rank_select_funds(
        data,
        cfg,
        inclusion_approach="top_n",
        n=2,
        score_by="Sortino",
        window_key=window_key,
        bundle=bundle,
    )

    stats_after_second = selector_cache_stats()
    assert stats_after_second["selector_cache_hits"] >= 1
    frame = bundle.metrics_frame()
    assert {"Sharpe", "Sortino"}.issubset(set(frame.columns))

    clear_window_metric_cache()
