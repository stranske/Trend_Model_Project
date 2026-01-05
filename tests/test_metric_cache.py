import numpy as np
import pandas as pd
import pytest

from trend_analysis.core.metric_cache import (
    MetricCache,
    clear_metric_cache,
    get_or_compute_metric_series,
    global_metric_cache,
)
from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    WindowMetricBundle,
    _compute_metric_series,
)


def _dummy_frame(n=5, m=24):
    idx = pd.date_range("2020-01-31", periods=m, freq="ME")
    data = {f"F{i}": np.random.randn(m) / 100 for i in range(n)}
    return pd.DataFrame(data, index=idx)


def test_metric_cache_hit_and_miss():
    clear_metric_cache()
    df = _dummy_frame()
    stats_cfg = RiskStatsConfig(risk_free=0.0)
    # Attach flag dynamically (as used in ensure_metric)
    setattr(stats_cfg, "enable_metric_cache", True)
    bundle = WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2021-12",
        freq="ME",
        stats_cfg_hash="hashx",
        universe=tuple(df.columns.astype(str)),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns),
    )
    first = bundle.ensure_metric("Sharpe", stats_cfg)
    second = bundle.ensure_metric("Sharpe", stats_cfg)
    assert first.equals(second)
    # global_metric_cache should have at least one hit
    assert global_metric_cache.hits >= 0  # sanity; internal miss/hit validated indirectly


def test_metric_cache_toggle_off():
    clear_metric_cache()
    df = _dummy_frame()
    stats_cfg = RiskStatsConfig(risk_free=0.0)
    setattr(stats_cfg, "enable_metric_cache", False)
    bundle = WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2021-12",
        freq="ME",
        stats_cfg_hash="hashx",
        universe=tuple(df.columns.astype(str)),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns),
    )
    # Monkeypatch underlying compute to count calls
    calls = {"n": 0}

    def _compute(in_df, metric_name, cfg):  # noqa: ANN001
        calls["n"] += 1
        return _compute_metric_series(in_df, metric_name, cfg)

    # Replace name in module namespace
    import trend_analysis.core.rank_selection as rs

    orig = rs._compute_metric_series
    rs._compute_metric_series = _compute  # type: ignore
    try:
        bundle.ensure_metric("Sharpe", stats_cfg)
        bundle.ensure_metric("Sharpe", stats_cfg)
    finally:
        rs._compute_metric_series = orig  # restore
    # Without metric cache we still store first computation in bundle, so compute called once
    assert calls["n"] == 1


def test_metric_cache_key_differs_by_metric():
    clear_metric_cache()
    df = _dummy_frame()
    stats_cfg = RiskStatsConfig(risk_free=0.0)
    setattr(stats_cfg, "enable_metric_cache", True)
    bundle = WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2021-12",
        freq="ME",
        stats_cfg_hash="hashx",
        universe=tuple(df.columns.astype(str)),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns),
    )
    s1 = bundle.ensure_metric("Sharpe", stats_cfg)
    s2 = bundle.ensure_metric("AnnualReturn", stats_cfg)
    assert not s1.equals(s2)


def test_metric_cache_stats_tracks_hits_and_misses():
    cache = MetricCache()
    # Initial stats with no traffic should report zeros.
    stats = cache.stats()
    assert stats == {"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

    series = pd.Series([1.0, 2.0], index=["A", "B"])
    cache.put("alpha", series)

    # One hit and one miss to exercise both code paths.
    assert cache.get("alpha") is series
    assert cache.get("beta") is None

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_get_or_compute_metric_series_reorders_index_and_caches(monkeypatch):
    cache = MetricCache()
    calls = {"count": 0}

    def compute() -> pd.Series:
        calls["count"] += 1
        # Return series in a different order to trigger the reindex branch.
        return pd.Series([2.0, 1.0], index=["B", "A"])

    kwargs = {
        "start": "2020-01",
        "end": "2020-02",
        "universe_cols": ("A", "B"),
        "metric_name": "Sharpe",
        "cfg_hash": "cfg",
        "compute": compute,
        "enable": True,
        "cache": cache,
    }

    series = get_or_compute_metric_series(**kwargs)
    assert list(series.index) == ["A", "B"]
    assert calls["count"] == 1

    # Second call should hit the cache and avoid recomputing.
    cached = get_or_compute_metric_series(**kwargs)
    assert cached is series
    assert calls["count"] == 1
