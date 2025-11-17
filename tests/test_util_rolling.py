import numpy as np
import pandas as pd

from trend_analysis.util.rolling import rolling_shifted


def test_rolling_shifted_mean_matches_manual() -> None:
    series = pd.Series(np.linspace(-0.05, 0.05, 10), index=pd.RangeIndex(10))
    result = rolling_shifted(series, window=3, agg="mean")
    expected = series.rolling(window=3).mean().shift(1)
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
