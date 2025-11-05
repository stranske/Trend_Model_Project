import numpy as np
import pandas as pd
import pytest

from trend_analysis.signals import TrendSpec, compute_trend_signals


def test_compute_trend_signals_vol_adjust_and_attrs():
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, 0.04, 0.02, 0.03, 0.05],
            "asset_b": [0.02, 0.01, 0.03, 0.07, 0.06],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    spec = TrendSpec(window=3, lag=1, vol_adjust=True, vol_target=0.5)

    signal = compute_trend_signals(returns, spec)

    numeric = returns.astype(float)
    rolling_mean = numeric.rolling(window=3, min_periods=3).mean().shift(1)
    rolling_std = numeric.rolling(window=3, min_periods=3).std(ddof=0).shift(1)
    expected = rolling_mean * (spec.vol_target / rolling_std)

    pd.testing.assert_frame_equal(signal, expected)
    assert signal.attrs["spec"]["window"] == 3
    assert signal.attrs["lag"] == 1
    assert signal.attrs["kind"] == "tsmom"


def test_compute_trend_signals_supports_row_zscore():
    returns = pd.DataFrame(
        {
            "a": [0.1, -0.2, 0.3, 0.4],
            "b": [0.2, -0.1, 0.1, 0.5],
            "c": [0.3, -0.3, 0.2, 0.6],
        }
    )
    spec = TrendSpec(window=2, lag=1, zscore=True)
    signal = compute_trend_signals(returns, spec)

    row_means = signal.mean(axis=1)
    assert np.allclose(row_means.fillna(0.0), 0.0)
    assert (signal.index == returns.index).all()


@pytest.mark.parametrize(
    "kwargs, error_message",
    [
        ({"window": 0}, "window"),
        ({"window": 5, "min_periods": 0}, "min_periods"),
        ({"window": 5, "lag": 0}, "lag"),
        ({"window": 5, "vol_target": -0.5}, "vol_target"),
    ],
)
def test_trend_spec_validates_inputs(kwargs, error_message):
    with pytest.raises(ValueError) as excinfo:
        TrendSpec(**kwargs)
    assert error_message in str(excinfo.value)


def test_compute_trend_signals_requires_non_empty_returns():
    with pytest.raises(ValueError, match="returns cannot be empty"):
        compute_trend_signals(pd.DataFrame(), TrendSpec(window=3))
