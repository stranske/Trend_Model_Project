"""Tests for rolling metrics module."""

import pandas as pd

from trend_analysis.metrics import rolling


def test_rolling_information_ratio_basic():
    """Rolling IR matches manual calculation."""

    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.005],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    benchmark = pd.Series(0.0, index=returns.index)

    result = rolling.rolling_information_ratio(returns, benchmark, window=2)

    excess = returns - benchmark
    expected = excess.rolling(2).mean() / excess.rolling(2).std(ddof=1)
    expected = expected.rename("rolling_ir")

    pd.testing.assert_series_equal(result, expected)


def test_rolling_information_ratio_scalar_benchmark():
    """Scalar benchmark is broadcast correctly."""

    returns = pd.Series([0.01, -0.02, 0.015, 0.005])
    result = rolling.rolling_information_ratio(returns, benchmark=0.005, window=2)

    assert isinstance(result, pd.Series)
    assert result.name == "rolling_ir"
    assert len(result) == len(returns)


def test_rolling_information_ratio_defaults_to_zero_benchmark():
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
