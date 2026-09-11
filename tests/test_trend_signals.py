import logging
from functools import partial

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from trend_analysis import pipeline, pipeline_helpers, signals
from trend_analysis.pipeline import compute_signal
from trend_analysis.signals import TrendSpec, compute_trend_signals


def _sample_returns(rows: int = 16) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=rows, freq="ME")
    data = {
        "fund_a": np.linspace(-0.02, 0.03, rows),
        "fund_b": np.cos(np.linspace(0.0, np.pi, rows)) * 0.02,
        "fund_c": np.sin(np.linspace(0.0, 2 * np.pi, rows)) * 0.015,
    }
    return pd.DataFrame(data, index=idx)


def test_trend_spec_window_changes_behaviour():
    returns = _sample_returns()
    fast = compute_trend_signals(returns, TrendSpec(window=3))
    slow = compute_trend_signals(returns, TrendSpec(window=6))
    # After the warm-up period the frames should differ when the window changes.
    fast_slice = fast.iloc[8:]
    slow_slice = slow.iloc[8:]
    assert not np.allclose(
        fast_slice.fillna(0.0).to_numpy(),
        slow_slice.fillna(0.0).to_numpy(),
    )


def test_vol_adjustment_changes_scale():
    returns = _sample_returns()
    base = compute_trend_signals(returns, TrendSpec(window=4))
    adjusted = compute_trend_signals(returns, TrendSpec(window=4, vol_adjust=True, vol_target=1.0))
    comparison = base.iloc[6:].fillna(0.0).to_numpy()
    adjusted_comp = adjusted.iloc[6:].fillna(0.0).to_numpy()
    assert not np.allclose(comparison, adjusted_comp)


def test_zscore_normalisation_rows_are_standardised():
    returns = _sample_returns()
    frame = compute_trend_signals(returns, TrendSpec(window=4, zscore=True))
    valid = frame.dropna(how="all").iloc[6:]
    if not valid.empty:
        mean = valid.mean(axis=1)
        std = valid.std(axis=1, ddof=0)
        assert np.allclose(mean.to_numpy(), np.zeros(mean.shape[0]), atol=1e-9)
        assert np.allclose(std.to_numpy(), np.ones(std.shape[0]), atol=1e-9)


def test_signal_is_shift_safe():
    returns = _sample_returns()
    spec = TrendSpec(window=4)
    baseline = compute_trend_signals(returns, spec)
    tweaked = returns.copy()
    tweaked.iloc[-1, 0] += 5.0
    shifted = compute_trend_signals(tweaked, spec)
    tm.assert_frame_equal(
        baseline.iloc[:-1],
        shifted.iloc[:-1],
        check_names=False,
    )


def test_pipeline_compute_signal_uses_trend_engine(monkeypatch):
    returns = _sample_returns()[["fund_a"]].rename(columns={"fund_a": "returns"})
    calls = []

    def record_engine(frame, spec):
        calls.append(spec)
        return compute_trend_signals(frame, spec)

    monkeypatch.setattr(signals, "compute_trend_signals", record_engine)
    monkeypatch.setattr(pipeline, "get_cache", lambda: None)
    series = compute_signal(returns, window=4)
    spec = TrendSpec(window=4, min_periods=4, lag=1)
    assert calls == [spec]
    expected = compute_trend_signals(returns, spec)["returns"].rename("returns_signal")
    tm.assert_series_equal(series, expected)


def test_compute_trend_signals_reuses_cached_numeric_frame() -> None:
    returns = _sample_returns()
    spec = TrendSpec(window=4, vol_adjust=True)

    compute_trend_signals(returns, spec)
    memo = returns.attrs.get("_trend_signal_cache")
    assert isinstance(memo, dict)
    cached_entry = memo.get("float_frame")
    assert cached_entry is not None
    cached_numeric = cached_entry.get() if hasattr(cached_entry, "get") else cached_entry
    assert isinstance(cached_numeric, pd.DataFrame)

    compute_trend_signals(returns, spec)
    memo_after = returns.attrs.get("_trend_signal_cache")
    assert memo_after
    cached_again = memo_after.get("float_frame")
    next_numeric = cached_again.get() if hasattr(cached_again, "get") else cached_again
    assert next_numeric is cached_numeric
    mean_key = ("rolling", "mean", spec.window, spec.window)
    mean_entry = memo_after.get(mean_key)
    assert mean_entry is not None
    cached_mean = mean_entry.get() if hasattr(mean_entry, "get") else mean_entry
    assert isinstance(cached_mean, pd.DataFrame)


def test_compute_trend_signals_logs_stage_timings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    returns = _sample_returns(rows=10)
    spec = TrendSpec(window=3, vol_adjust=True, vol_target=1.0, zscore=True)

    with caplog.at_level(logging.DEBUG, logger="trend_analysis.signals"):
        compute_trend_signals(returns, spec)

    stage_logs = [rec.message for rec in caplog.records if "compute_trend_signals[" in rec.message]
    assert stage_logs, "expected timing logs for compute_trend_signals"
    assert any("float_coerce" in msg for msg in stage_logs)


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("min_periods", [None, 2])
def test_pipeline_signal_multicolumn_equivalence(monkeypatch, lag, min_periods):
    returns = _sample_returns()
    returns.loc[returns.index[2], "fund_b"] = np.nan
    original = returns.copy(deep=True)
    # Keep the public helper signature stable while exercising a different spec lag.
    monkeypatch.setattr(pipeline_helpers, "TrendSpec", partial(TrendSpec, lag=lag))
    monkeypatch.setattr(pipeline, "get_cache", lambda: None)
    expected = compute_trend_signals(
        returns.copy(), TrendSpec(window=4, min_periods=min_periods, lag=lag)
    )
    for column in returns.columns:
        actual = compute_signal(returns, column=column, window=4, min_periods=min_periods)
        tm.assert_series_equal(actual, expected[column].rename(f"{column}_signal"))
    tm.assert_frame_equal(returns, original)


def test_pipeline_signal_cache_separates_spec_lags(monkeypatch):
    class MemoryCache:
        def __init__(self):
            self.values = {}

        def is_enabled(self):
            return True

        def get_or_compute(self, dataset, window, frequency, method, compute):
            key = (dataset, window, frequency, method)
            if key not in self.values:
                self.values[key] = compute()
            return self.values[key]

    cache = MemoryCache()
    monkeypatch.setattr(pipeline, "get_cache", lambda: cache)
    returns = _sample_returns()
    results = []
    for lag in [1, 2]:
        monkeypatch.setattr(pipeline_helpers, "TrendSpec", partial(TrendSpec, lag=lag))
        results.append(compute_signal(returns, column="fund_a", window=4))
    assert len(cache.values) == 2
    assert not results[0].equals(results[1])
    assert compute_signal(returns, column="fund_a", window=4) is results[1]


def test_pipeline_signal_preserves_empty_series():
    returns = pd.DataFrame({"returns": pd.Series(dtype=float)})
    expected = pd.Series(dtype=float, name="returns_signal")
    tm.assert_series_equal(compute_signal(returns), expected)


def test_pipeline_signal_ignores_inherited_engine_memo(monkeypatch):
    """A frame derived from one the engine has already seen must not reuse its memo.

    ``pandas`` copies ``attrs`` through ``df[col]``/``astype``/``to_frame``, so
    without an explicit clear the adapter hands the engine a memo whose cached
    ``float_frame`` and rolling frames belong to the caller's *earlier* values.
    """

    monkeypatch.setattr(pipeline, "get_cache", lambda: None)
    seeded = pd.DataFrame({"returns": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    compute_trend_signals(seeded, TrendSpec(window=3, min_periods=3))
    memo_attr = signals._MEMO_ATTR
    assert memo_attr in seeded.attrs, "precondition: the engine memoised the seed frame"

    derived = seeded * 100.0
    assert memo_attr in derived.attrs, "precondition: pandas propagated the memo"

    actual = compute_signal(derived, column="returns", window=3, min_periods=3)
    clean = pd.DataFrame({"returns": derived["returns"].to_numpy()}, index=derived.index)
    expected = compute_signal(clean, column="returns", window=3, min_periods=3)
    tm.assert_series_equal(actual, expected)
    # The caller's own frames keep their memo; only the adapter's copy is cleared.
    assert memo_attr in derived.attrs
    assert memo_attr in seeded.attrs


def test_clear_signal_cache_leaves_the_source_frame_untouched():
    seeded = pd.DataFrame({"returns": [1.0, 2.0, 3.0, 4.0]})
    compute_trend_signals(seeded, TrendSpec(window=2, min_periods=2))
    derived = seeded["returns"].to_frame()

    assert signals.clear_signal_cache(derived) is derived
    assert signals._MEMO_ATTR not in derived.attrs
    assert signals._MEMO_ATTR in seeded.attrs
