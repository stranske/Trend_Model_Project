import pandas as pd
import pandas.testing as tm

from trend_analysis.perf.rolling_cache import RollingCache, compute_dataset_hash


def test_compute_dataset_hash_reflects_data_changes():
    series_a = pd.Series([1.0, 2.0, 3.0], name="value")
    series_b = pd.Series([1.0, 2.0, 4.0], name="value")

    hash_a = compute_dataset_hash([series_a])
    hash_b = compute_dataset_hash([series_b])

    assert hash_a != hash_b


def test_rolling_cache_persists_results(tmp_path):
    cache = RollingCache(cache_dir=tmp_path)
    dataset_series = pd.Series([0.1, 0.2, 0.3], name="signal")
    dataset_hash = compute_dataset_hash([dataset_series])

    calls: list[pd.Series] = []

    def compute() -> pd.Series:
        result = pd.Series([0.1, 0.2, 0.3], name="signal")
        calls.append(result)
        return result

    first = cache.get_or_compute(dataset_hash, 3, "M", "rolling_mean", compute)
    second = cache.get_or_compute(dataset_hash, 3, "M", "rolling_mean", compute)

    assert len(calls) == 1
    tm.assert_series_equal(first, second)


def test_rolling_cache_respects_disable(tmp_path):
    cache = RollingCache(cache_dir=tmp_path)
    cache.set_enabled(False)
    dataset_series = pd.Series([5.0, 6.0, 7.0], name="value")
    dataset_hash = compute_dataset_hash([dataset_series])

    counter = {"calls": 0}

    def compute() -> pd.Series:
        counter["calls"] += 1
        return pd.Series([5.0, 6.0, 7.0], name="value")

    cache.get_or_compute(dataset_hash, 2, "unknown", "rolling_mean", compute)
    cache.get_or_compute(dataset_hash, 2, "unknown", "rolling_mean", compute)

    assert counter["calls"] == 2


def test_rolling_cache_invalidation_on_param_change(tmp_path):
    """Changing window or method should trigger recomputation (cache miss).

    Acceptance (Issue #1440): cache invalidates on parameter change. We assert
    that:
        * same (hash, window, freq, method) -> single compute
        * different window -> recompute
        * same window but different method -> recompute
    """
    cache = RollingCache(cache_dir=tmp_path)
    series = pd.Series([0.1, 0.2, 0.3, 0.4], name="sig")
    dataset_hash = compute_dataset_hash([series])

    calls = {"n": 0}

    def compute() -> pd.Series:  # pragma: no cover - trivial
        calls["n"] += 1
        return series

    # First call (window=3, method=a) -> miss
    cache.get_or_compute(dataset_hash, 3, "M", "method_a", compute)
    # Second identical call -> hit
    cache.get_or_compute(dataset_hash, 3, "M", "method_a", compute)
    assert calls["n"] == 1
    # Change window -> miss
    cache.get_or_compute(dataset_hash, 4, "M", "method_a", compute)
    assert calls["n"] == 2
    # Same window but different method -> miss
    cache.get_or_compute(dataset_hash, 4, "M", "method_b", compute)
    assert calls["n"] == 3


def test_rolling_cache_emits_timing_logs(monkeypatch, tmp_path):
    cache = RollingCache(cache_dir=tmp_path)
    series = pd.Series([0.1, 0.2, 0.3], name="sig")
    dataset_hash = compute_dataset_hash([series])

    logged: list[tuple[str, dict[str, object]]] = []

    def fake_log(stage: str, **fields: object) -> None:
        logged.append((stage, fields))

    monkeypatch.setattr(
        "trend_analysis.perf.rolling_cache.log_timing",
        fake_log,
    )

    cache.get_or_compute(dataset_hash, 3, "M", "method_a", lambda: series)
    cache.get_or_compute(dataset_hash, 3, "M", "method_a", lambda: series)

    assert [entry[1]["status"] for entry in logged] == ["miss", "hit"]
