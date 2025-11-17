from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.signals import TrendSpec, compute_trend_signals


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=8, freq="M")
    data = {
        "AssetA": np.linspace(-0.02, 0.03, len(index)),
        "AssetB": np.cos(np.linspace(0.0, np.pi, len(index))) * 0.02,
    }
    return pd.DataFrame(data, index=index)


def test_compute_trend_signals_simple_mean(sample_returns: pd.DataFrame) -> None:
    spec = TrendSpec(window=2, lag=1)
    frame = compute_trend_signals(sample_returns, spec)
    expected = sample_returns.rolling(2).mean().shift(1)
    pd.testing.assert_frame_equal(frame, expected)


def test_vol_adjustment_scales_output(sample_returns: pd.DataFrame) -> None:
    base = compute_trend_signals(sample_returns, TrendSpec(window=3, lag=1))
    adjusted = compute_trend_signals(
        sample_returns, TrendSpec(window=3, lag=1, vol_adjust=True, vol_target=1.0)
    )
    base_slice = base.iloc[4:].fillna(0.0).to_numpy()
    adjusted_slice = adjusted.iloc[4:].fillna(0.0).to_numpy()
    assert not np.allclose(base_slice, adjusted_slice)


def test_zscore_rows_standardised(sample_returns: pd.DataFrame) -> None:
    spec = TrendSpec(window=3, lag=1, zscore=True)
    frame = compute_trend_signals(sample_returns, spec)
    valid = frame.dropna(how="all").iloc[4:]
    if not valid.empty:
        mean = valid.mean(axis=1)
        std = valid.std(axis=1, ddof=0)
        assert np.allclose(mean.to_numpy(), np.zeros(len(mean)), atol=1e-9)
        assert np.allclose(std.to_numpy(), np.ones(len(std)), atol=1e-9)


def test_lag_enforces_shift(sample_returns: pd.DataFrame) -> None:
    spec = TrendSpec(window=3, lag=2)
    frame = compute_trend_signals(sample_returns, spec)
    expected = sample_returns.rolling(3).mean().shift(2)
    pd.testing.assert_frame_equal(frame, expected)


def test_window_parameter_changes_behaviour(sample_returns: pd.DataFrame) -> None:
    short = compute_trend_signals(sample_returns, TrendSpec(window=2, lag=1))
    long = compute_trend_signals(sample_returns, TrendSpec(window=5, lag=1))
    comparison = short.iloc[6:].fillna(0.0).to_numpy()
    long_comp = long.iloc[6:].fillna(0.0).to_numpy()
    assert not np.allclose(comparison, long_comp)


def test_compute_trend_signals_never_uses_same_day(
    sample_returns: pd.DataFrame,
) -> None:
    spec = TrendSpec(window=3, lag=1, vol_adjust=True, vol_target=1.0)
    baseline = compute_trend_signals(sample_returns, spec)
    tweaked = sample_returns.copy()
    tweaked.iloc[-1] += 10.0
    shifted = compute_trend_signals(tweaked, spec)
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1],
        shifted.iloc[:-1],
        check_names=False,
    )
