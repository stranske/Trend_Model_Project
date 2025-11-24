import math

import pandas as pd
import pandas.testing as tm
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trend_analysis.pipeline import compute_signal, position_from_signal


def test_compute_signal_matches_trailing_mean_without_current_row():
    df = pd.DataFrame({"returns": [0.01, 0.03, 0.02, -0.01, 0.05]})
    signal = compute_signal(df, window=3)
    # Causal spec: signal equals trailing mean excluding current row (rolling mean then shift)
    expected = df["returns"].rolling(window=3, min_periods=3).mean().shift(1)
    expected.name = signal.name
    tm.assert_series_equal(signal, expected)


def test_positions_ignore_future_data():
    base = pd.DataFrame({"returns": [0.02, -0.01, 0.03, -0.02, 0.04, 0.01]})
    original_positions = position_from_signal(compute_signal(base, window=3))

    tweaked = base.copy()
    tweaked.loc[tweaked.index[-1], "returns"] = 10.0
    tweaked_positions = position_from_signal(compute_signal(tweaked, window=3))

    tm.assert_series_equal(
        original_positions.iloc[:-1],
        tweaked_positions.iloc[:-1],
        check_names=False,
    )


@given(
    st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=32,
    )
)
@settings(max_examples=50, deadline=None)
def test_shift_safe_pipeline_is_causal(returns):
    df = pd.DataFrame({"returns": returns})
    global_positions = position_from_signal(compute_signal(df, window=3))

    for idx in range(len(df)):
        partial = df.iloc[: idx + 1]
        partial_positions = position_from_signal(compute_signal(partial, window=3))
        assert float(global_positions.iloc[idx]) == float(partial_positions.iloc[-1])


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=48,
    )
)
@settings(max_examples=50, deadline=None)
def test_compute_signal_only_uses_past_data(returns):
    df = pd.DataFrame({"returns": returns})
    window = 3
    signal = compute_signal(df, window=window)

    for idx, value in enumerate(signal.to_numpy()):
        history = df["returns"].iloc[max(0, idx - window) : idx]
        if len(history) < window:
            assert math.isnan(value)
            continue
        expected = history.mean()
        assert value == pytest.approx(expected)
