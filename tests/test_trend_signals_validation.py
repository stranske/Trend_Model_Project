"""Focused regression coverage for :mod:`trend_analysis.signals`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.signals import TrendSpec, compute_trend_signals


def test_trend_spec_validates_parameters() -> None:
    """Invalid ``TrendSpec`` inputs should raise descriptive errors."""

    with pytest.raises(ValueError, match="window must be a positive"):
        TrendSpec(window=0)
    with pytest.raises(ValueError, match="min_periods must be positive"):
        TrendSpec(min_periods=0)
    with pytest.raises(ValueError, match="lag must be at least 1"):
        TrendSpec(lag=0)
    with pytest.raises(ValueError, match="vol_target must be non-negative"):
        TrendSpec(vol_target=-0.5)


def test_compute_trend_signals_rejects_empty_returns() -> None:
    """An empty input frame should fail fast with a clear error."""

    with pytest.raises(ValueError, match="returns cannot be empty"):
        compute_trend_signals(pd.DataFrame(), TrendSpec())


def test_vol_adjust_without_target_inverts_rolling_std() -> None:
    """When ``vol_target`` is omitted the inverse rolling std should be used."""

    index = pd.date_range("2024-01-31", periods=8, freq="M")
    base_data = {
        "fund_a": np.linspace(-0.01, 0.03, len(index)),
        "fund_b": np.linspace(0.015, -0.02, len(index)),
    }
    returns = pd.DataFrame(base_data, index=index)

    spec_no_target = TrendSpec(window=3, min_periods=3, vol_adjust=True)
    adjusted = compute_trend_signals(returns, spec_no_target)

    baseline_spec = TrendSpec(window=3, min_periods=3)
    baseline = compute_trend_signals(returns, baseline_spec)

    rolling_std = (
        returns.astype(float)
        .rolling(window=3, min_periods=3)
        .std(ddof=0)
        .shift(spec_no_target.lag)
    )
    expected = baseline.mul(1.0 / rolling_std)

    pd.testing.assert_frame_equal(adjusted.iloc[3:], expected.iloc[3:])
    assert adjusted.attrs["spec"]["vol_adjust"] is True
