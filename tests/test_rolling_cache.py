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
