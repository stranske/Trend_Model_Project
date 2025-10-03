from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.signals import SignalFrame, TrendSpec, generate_signals


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = {
        "AssetA": [100, 102, 104, 108, 110, 115],
        "AssetB": [50, 49, 51, 50, 52, 55],
    }
    return pd.DataFrame(data, index=index)


def test_generate_signals_basic_momentum(sample_prices: pd.DataFrame) -> None:
    spec = TrendSpec(lookback=1, execution_lag=0)
    signal = generate_signals(sample_prices, spec)

    expected = sample_prices.pct_change(1)
    expected.columns.name = "asset"
    pd.testing.assert_frame_equal(signal.stage("raw_momentum"), expected)
    pd.testing.assert_frame_equal(signal.stage("vol_adjusted"), expected)
    pd.testing.assert_frame_equal(signal.stage("normalized"), expected)
    pd.testing.assert_frame_equal(signal.final, expected)


def test_generate_signals_vol_adjust_and_zscore(sample_prices: pd.DataFrame) -> None:
    spec = TrendSpec(
        lookback=1,
        vol_lookback=2,
        use_vol_adjust=True,
        use_zscore=True,
        execution_lag=1,
    )
    frame = generate_signals(sample_prices, spec, rebalance_dates=sample_prices.index)

    momentum = sample_prices.pct_change(1)
    realised_vol = sample_prices.pct_change().rolling(2).std().replace(0.0, np.nan)
    expected_vol_adjusted = momentum.divide(realised_vol)
    expected_vol_adjusted.columns.name = "asset"

    demeaned = expected_vol_adjusted.sub(
        expected_vol_adjusted.mean(axis=1, skipna=True), axis=0
    )
    std = expected_vol_adjusted.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    expected_normalized = demeaned.divide(std, axis=0)
    expected_normalized = expected_normalized.where(np.isfinite(expected_normalized))
    zero_std_rows = std.isna() & expected_vol_adjusted.notna().any(axis=1)
    if zero_std_rows.any():
        expected_normalized.loc[zero_std_rows] = 0.0
        expected_normalized = expected_normalized.where(expected_vol_adjusted.notna())
    expected_normalized.columns.name = "asset"

    expected_signal = expected_normalized.shift(1)
    expected_signal.columns.name = "asset"

    pd.testing.assert_frame_equal(frame.stage("vol_adjusted"), expected_vol_adjusted)
    pd.testing.assert_frame_equal(frame.stage("normalized"), expected_normalized)
    pd.testing.assert_frame_equal(frame.final, expected_signal)


def test_signal_frame_columns_are_consistent(sample_prices: pd.DataFrame) -> None:
    base_spec = TrendSpec(lookback=1, execution_lag=0)
    vol_spec = TrendSpec(lookback=1, vol_lookback=2, use_vol_adjust=True, execution_lag=0)
    zscore_spec = TrendSpec(lookback=1, use_zscore=True, execution_lag=0)

    base_frame = generate_signals(sample_prices, base_spec).frame
    vol_frame = generate_signals(sample_prices, vol_spec).frame
    zscore_frame = generate_signals(sample_prices, zscore_spec).frame
    frames = [base_frame, vol_frame, zscore_frame]

    for frame in frames:
        assert isinstance(frame, pd.DataFrame)
        assert frame.columns.names == ["stage", "asset"]
        assert tuple(frame.columns.get_level_values(0).unique()) == SignalFrame._STAGES

    first_columns = frames[0].columns.tolist()
    for frame in frames[1:]:
        assert frame.columns.tolist() == first_columns


def test_execution_lag_applied_only_on_rebalance_dates(sample_prices: pd.DataFrame) -> None:
    spec = TrendSpec(lookback=1, execution_lag=1)
    rebalance_dates = sample_prices.index[2:]
    frame = generate_signals(sample_prices, spec, rebalance_dates=rebalance_dates)

    normalized = frame.stage("normalized")
    expected = normalized.copy()
    shifted = normalized.shift(1)
    expected.loc[rebalance_dates] = shifted.loc[rebalance_dates]

    pd.testing.assert_frame_equal(frame.final, expected)


def test_execution_lag_without_rebalance_dates_shifts_all_periods(
    sample_prices: pd.DataFrame,
) -> None:
    spec = TrendSpec(lookback=1, execution_lag=1)
    frame = generate_signals(sample_prices, spec, rebalance_dates=None)

    normalized = frame.stage("normalized")
    expected = normalized.shift(spec.execution_lag)

    pd.testing.assert_frame_equal(frame.final, expected)


def test_execution_lag_errors_when_rebalance_dates_missing_from_index(
    sample_prices: pd.DataFrame,
) -> None:
    spec = TrendSpec(lookback=1, execution_lag=1)
    missing_dates = pd.date_range("2019-12-01", periods=3, freq="D")

    with pytest.raises(ValueError, match="execution lag cannot be applied"):
        generate_signals(sample_prices, spec, rebalance_dates=missing_dates)


def test_trend_spec_toggles_modify_signal(sample_prices: pd.DataFrame) -> None:
    base_spec = TrendSpec(lookback=1, execution_lag=0)
    vol_spec = TrendSpec(lookback=1, vol_lookback=2, use_vol_adjust=True, execution_lag=0)
    zscore_spec = TrendSpec(lookback=1, use_zscore=True, execution_lag=0)

    base_frame = generate_signals(sample_prices, base_spec)
    vol_frame = generate_signals(sample_prices, vol_spec)
    zscore_frame = generate_signals(sample_prices, zscore_spec)

    assert not base_frame.final.equals(vol_frame.final)
    assert not vol_frame.final.equals(zscore_frame.final)
