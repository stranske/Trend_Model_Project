"""Validation and error handling tests for :mod:`trend_analysis.signals`."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.signals import TrendSpec, compute_trend_signals


def test_trend_spec_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        TrendSpec(window=0)


def test_trend_spec_rejects_invalid_min_periods() -> None:
    with pytest.raises(ValueError, match="min_periods must be positive"):
        TrendSpec(min_periods=0)


def test_trend_spec_rejects_invalid_lag() -> None:
    with pytest.raises(ValueError, match="lag must be at least 1"):
        TrendSpec(lag=0)


def test_trend_spec_rejects_negative_vol_target() -> None:
    with pytest.raises(ValueError, match="vol_target must be non-negative"):
        TrendSpec(vol_target=-0.1)


def test_compute_trend_signals_requires_non_empty_returns() -> None:
    empty = pd.DataFrame(columns=["FundA"], dtype=float)
    with pytest.raises(ValueError, match="returns cannot be empty"):
        compute_trend_signals(empty, TrendSpec())


def test_compute_trend_signals_vol_adjust_without_target() -> None:
    data = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.03, 0.01, 0.0, -0.01, -0.02],
            "FundB": [0.0, 0.01, 0.015, -0.005, -0.01, 0.0, 0.005],
        },
        index=pd.date_range("2024-01-31", periods=7, freq="M"),
    )
    spec = TrendSpec(window=3, vol_adjust=True, vol_target=None)

    result = compute_trend_signals(data, spec)

    assert result.shape == data.shape
    assert not result.isna().all().all(), "vol adjustment should produce finite values"
    assert result.attrs["spec"]["vol_adjust"] is True
