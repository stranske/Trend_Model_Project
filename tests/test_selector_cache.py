import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    clear_window_metric_cache,
    get_window_metric_bundle,
    make_window_key,
    rank_select_funds,
    selector_cache_scope,
    selector_cache_stats,
    set_window_metric_cache_limit,
)


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


def test_selector_cache_eviction_respects_limit() -> None:
    clear_window_metric_cache()
    previous_limit = set_window_metric_cache_limit(1)
    try:
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        data = pd.DataFrame(
            {
                "Fund A": np.linspace(0.01, 0.06, len(idx)),
                "Fund B": np.linspace(0.015, 0.065, len(idx)),
                "Fund C": np.linspace(0.02, 0.07, len(idx)),
            },
            index=idx,
        )
        cfg = RiskStatsConfig()
        key_one = make_window_key("2020-01", "2020-06", ("Fund A", "Fund B"), cfg)
        key_two = make_window_key("2020-02", "2020-07", ("Fund B", "Fund C"), cfg)

        rank_select_funds(
            data[["Fund A", "Fund B"]],
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            window_key=key_one,
        )
        rank_select_funds(
            data[["Fund B", "Fund C"]],
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            window_key=key_two,
        )

        assert get_window_metric_bundle(key_one) is None
        assert get_window_metric_bundle(key_two) is not None
        stats = selector_cache_stats()
        assert stats["entries"] <= 1
    finally:
        set_window_metric_cache_limit(previous_limit)
        clear_window_metric_cache()


def test_selector_cache_scope_isolation() -> None:
    clear_window_metric_cache()
    cfg = RiskStatsConfig()
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    data = pd.DataFrame(
        {"Fund A": np.linspace(0.01, 0.04, len(idx)), "Fund B": np.linspace(0.02, 0.05, len(idx))},
        index=idx,
    )

    key_a = make_window_key("2020-01", "2020-04", data.columns, cfg)
    key_b = make_window_key("2020-02", "2020-05", data.columns, cfg)

    with selector_cache_scope("run-a"):
        rank_select_funds(
            data,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            window_key=key_a,
        )
        assert selector_cache_stats()["entries"] == 1

    with selector_cache_scope("run-b"):
        assert selector_cache_stats()["entries"] == 0
        rank_select_funds(
            data,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            window_key=key_b,
        )
        assert selector_cache_stats()["entries"] == 1

    with selector_cache_scope("run-a"):
        assert get_window_metric_bundle(key_a) is not None
        assert get_window_metric_bundle(key_b) is None

    with selector_cache_scope("run-b"):
        assert get_window_metric_bundle(key_b) is not None
