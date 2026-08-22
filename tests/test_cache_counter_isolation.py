"""Isolation tests for selector cache instrumentation counters."""

import trend_analysis.core.rank_selection as rs
from trend_analysis.core.rank_selection import WindowMetricCache


def test_stats_track_active_cache_instance(monkeypatch):

    cache_a = WindowMetricCache()
    cache_a.record_hit()
    cache_a.record_hit()
    cache_a.record_miss()

    cache_b = WindowMetricCache()  # independent, untouched

    monkeypatch.setattr(rs, "_WINDOW_METRIC_CACHE", cache_a)
    assert rs.selector_cache_stats()["selector_cache_hits"] == 2
    assert rs.selector_cache_stats()["selector_cache_misses"] == 1

    # Switching to an independent instance must NOT leak cache_a's counts.
    monkeypatch.setattr(rs, "_WINDOW_METRIC_CACHE", cache_b)
    assert rs.selector_cache_stats()["selector_cache_hits"] == 0
    assert rs.selector_cache_stats()["selector_cache_misses"] == 0


def test_two_instances_do_not_share_counts():
    """Each :class:`WindowMetricCache` owns its own hit/miss counters."""

    cache_a = WindowMetricCache()
    cache_b = WindowMetricCache()

    cache_a.record_hit()
    cache_b.record_miss()
    cache_b.record_miss()

    assert cache_a.stats()["selector_cache_hits"] == 1
    assert cache_a.stats()["selector_cache_misses"] == 0
    assert cache_b.stats()["selector_cache_hits"] == 0
    assert cache_b.stats()["selector_cache_misses"] == 2


def test_clear_resets_counters_to_zero():
    """Clearing the cache zeroes the derived views (no stale module state)."""

    cache = WindowMetricCache()
    cache.record_hit()
    cache.record_miss()
    cache.clear()

    assert cache.stats()["selector_cache_hits"] == 0
    assert cache.stats()["selector_cache_misses"] == 0
