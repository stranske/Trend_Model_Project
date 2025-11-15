"""Property-based tests guarding against look-ahead leaks in signal helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as tm
from hypothesis import given, settings
from hypothesis import strategies as st

from trend_analysis.pipeline import compute_signal, position_from_signal


def _random_returns(
    min_size: int = 6, max_size: int = 40
) -> st.SearchStrategy[list[float]]:
    return st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


@given(_random_returns(), st.integers(min_value=1, max_value=6))
@settings(max_examples=75)
def test_compute_signal_is_strictly_causal(returns: list[float], window: int) -> None:
    df = pd.DataFrame({"returns": returns})
    signal = compute_signal(df, window=window)

    expected = df["returns"].rolling(window=window, min_periods=window).mean().shift(1)
    expected.name = signal.name

    tm.assert_series_equal(signal, expected)


@given(_random_returns(min_size=8, max_size=32), st.integers(min_value=2, max_value=6))
@settings(max_examples=60)
def test_intentional_leak_is_detected(returns: list[float], window: int) -> None:
    df = pd.DataFrame({"returns": returns})
    signal = compute_signal(df, window=window)

    leaking = df["returns"].rolling(window=window, min_periods=window).mean()
    leaking.name = signal.name

    idx = window - 1
    if idx < len(signal):
        assert math.isnan(signal.iloc[idx])
        assert not math.isnan(leaking.iloc[idx])

    tm.assert_series_equal(signal.iloc[window:], leaking.shift(1).iloc[window:])


@given(_random_returns(min_size=10, max_size=48))
@settings(max_examples=50)
def test_positions_depend_only_on_realised_history(returns: list[float]) -> None:
    df = pd.DataFrame({"returns": returns})
    signal = compute_signal(df, window=3)
    global_positions = position_from_signal(signal)

    for cutoff in range(1, len(df) + 1):
        prefix = df.iloc[:cutoff]
        prefix_signal = compute_signal(prefix, window=3)
        prefix_positions = position_from_signal(prefix_signal)
        assert np.isclose(
            float(global_positions.iloc[cutoff - 1]),
            float(prefix_positions.iloc[-1]),
        )


def test_future_modification_does_not_change_history() -> None:
    base = pd.DataFrame({"returns": [0.02, -0.01, 0.03, -0.02, 0.04, 0.01]})
    signal = compute_signal(base, window=3)
    positions = position_from_signal(signal)

    tweaked = base.copy()
    tweaked.loc[tweaked.index[-1], "returns"] = 10.0

    tweaked_signal = compute_signal(tweaked, window=3)
    tweaked_positions = position_from_signal(tweaked_signal)

    tm.assert_series_equal(
        signal.iloc[:-1], tweaked_signal.iloc[:-1], check_names=False
    )
    tm.assert_series_equal(
        positions.iloc[:-1],
        tweaked_positions.iloc[:-1],
        check_names=False,
    )
