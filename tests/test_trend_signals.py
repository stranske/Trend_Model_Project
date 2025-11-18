import logging

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

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
    adjusted = compute_trend_signals(
        returns, TrendSpec(window=4, vol_adjust=True, vol_target=1.0)
    )
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


def test_pipeline_compute_signal_uses_trend_engine():
    returns = _sample_returns()[["fund_a"]].rename(columns={"fund_a": "returns"})
    series = compute_signal(returns, window=4)
    spec = TrendSpec(window=4, min_periods=4)
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
    cached_numeric = (
        cached_entry.get() if hasattr(cached_entry, "get") else cached_entry
    )
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

    stage_logs = [
        rec.message for rec in caplog.records if "compute_trend_signals[" in rec.message
    ]
    assert stage_logs, "expected timing logs for compute_trend_signals"
    assert any("float_coerce" in msg for msg in stage_logs)
