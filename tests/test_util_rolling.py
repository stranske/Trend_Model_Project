import numpy as np
import pandas as pd

from trend_analysis.util.rolling import rolling_shifted


def test_rolling_shifted_mean_matches_manual() -> None:
    series = pd.Series(np.linspace(-0.05, 0.05, 10), index=pd.RangeIndex(10))
    result = rolling_shifted(series, window=3, agg="mean")
    expected = series.shift(1).rolling(window=3, min_periods=3).mean()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_shifted_callable_excludes_same_day() -> None:
    series = pd.Series(np.arange(1.0, 7.0), index=pd.RangeIndex(6))

    def tail_value(window: pd.Series) -> float:
        return float(window.iloc[-1])

    baseline = rolling_shifted(series, window=3, agg=tail_value)
    tweaked = series.copy()
    tweaked.iloc[-1] = 999.0
    shifted = rolling_shifted(tweaked, window=3, agg=tail_value)
    pd.testing.assert_series_equal(baseline.iloc[:-1], shifted.iloc[:-1])


def test_rolling_shifted_dataframe_sum_matches_manual() -> None:
    frame = pd.DataFrame(
        {
            "asset_a": np.linspace(0.0, 0.05, 6),
            "asset_b": np.linspace(-0.02, 0.03, 6),
        },
        index=pd.RangeIndex(6),
    )

    result = rolling_shifted(frame, window=2, agg="sum", min_periods=1)
    expected = frame.shift(1).rolling(window=2, min_periods=1).sum()
    pd.testing.assert_frame_equal(result, expected)


def test_rolling_shifted_std_respects_min_periods_and_is_causal() -> None:
    series = pd.Series(np.linspace(-0.1, 0.08, 12), index=pd.RangeIndex(12))

    baseline = rolling_shifted(series, window=4, agg="std", min_periods=2)
    tweaked = series.copy()
    tweaked.iloc[-1] = 999.0
    shifted = rolling_shifted(tweaked, window=4, agg="std", min_periods=2)

    expected = series.shift(1).rolling(window=4, min_periods=2).std(ddof=0)
    pd.testing.assert_series_equal(baseline, expected)
    pd.testing.assert_series_equal(baseline.iloc[:-1], shifted.iloc[:-1])
