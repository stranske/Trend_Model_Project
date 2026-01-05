"""Tests for the rolling metrics helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.metrics import rolling


def test_rolling_information_ratio_basic() -> None:
    """Rolling IR matches manual calculation."""

    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.005],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    benchmark = pd.Series(0.0, index=returns.index)

    result = rolling.rolling_information_ratio(returns, benchmark, window=2)

    excess = returns - benchmark
    expected = excess.rolling(2).mean() / excess.rolling(2).std(ddof=1)
    pd.testing.assert_series_equal(result, expected.rename("rolling_ir"))


def test_rolling_information_ratio_scalar_benchmark() -> None:
    """Scalar benchmark is broadcast correctly."""

    returns = pd.Series([0.01, -0.02, 0.015, 0.005])
    result = rolling.rolling_information_ratio(returns, benchmark=0.005, window=2)

    assert isinstance(result, pd.Series)
    assert result.name == "rolling_ir"
    assert len(result) == len(returns)


def test_rolling_information_ratio_defaults_to_zero_benchmark() -> None:
    """Passing ``None`` for the benchmark uses a zero baseline."""

    returns = pd.Series(
        [0.02, 0.01, -0.03, 0.015],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )

    result = rolling.rolling_information_ratio(returns, benchmark=None, window=2)

    zero_benchmark = pd.Series(0.0, index=returns.index)
    expected = (returns - zero_benchmark).rolling(2).mean() / (
        (returns - zero_benchmark).rolling(2).std(ddof=1)
    )
    pd.testing.assert_series_equal(result, expected.rename("rolling_ir"))


def test_rolling_information_ratio_uses_cache_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caching path should delegate to ``get_or_compute`` with rich tags."""

    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01],
        index=pd.date_range("2021-01-31", periods=4, freq="ME"),
        name="returns",
    )
    benchmark = pd.Series(
        [0.005, -0.01, 0.015, -0.002],
        index=returns.index,
        name="bench",
    )

    recorded: dict[str, object] = {}

    class DummyCache:
        def is_enabled(self) -> bool:
            return True

        def get_or_compute(self, dataset_hash, window, freq_tag, method_tag, compute_fn):
            recorded["dataset_hash"] = dataset_hash
            recorded["window"] = window
            recorded["freq_tag"] = freq_tag
            recorded["method_tag"] = method_tag
            result = compute_fn()
            recorded["result"] = result
            return result

    dummy = DummyCache()
    monkeypatch.setattr(rolling, "get_cache", lambda: dummy)

    result = rolling.rolling_information_ratio(returns, benchmark=benchmark, window=3)

    expected_bench = benchmark.reindex_like(returns).fillna(0.0)
    expected_hash = rolling.compute_dataset_hash([returns, expected_bench])

    assert recorded["dataset_hash"] == expected_hash
    assert recorded["window"] == 3
    assert recorded["freq_tag"] == str(returns.index.freqstr)
    assert recorded["method_tag"] == "rolling_information_ratio_ddof1"
    pd.testing.assert_series_equal(result, recorded["result"])


def test_rolling_information_ratio_cache_handles_unknown_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When frequency cannot be inferred the cache should fall back to
    ``unknown``."""

    returns = pd.Series(
        [0.02, -0.01, 0.015, -0.005],
        index=pd.to_datetime(["2021-01-31", "2021-02-02", "2021-03-15", "2021-03-29"]),
    )

    recorded: dict[str, object] = {}

    class DummyCache:
        def is_enabled(self) -> bool:
            return True

        def get_or_compute(self, dataset_hash, window, freq_tag, method_tag, compute_fn):
            recorded["freq_tag"] = freq_tag
            recorded["result"] = compute_fn()
            return recorded["result"]

    monkeypatch.setattr(rolling, "get_cache", lambda: DummyCache())

    result = rolling.rolling_information_ratio(returns, benchmark=0.0, window=2)

    assert recorded["freq_tag"] == "unknown"
    pd.testing.assert_series_equal(result, recorded["result"])
