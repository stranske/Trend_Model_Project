import pandas as pd
import pytest

from trend_analysis.metrics.rolling import rolling_information_ratio
from trend_analysis.perf.rolling_cache import get_cache


def test_rolling_information_ratio_without_cache():
    cache = get_cache()
    original = cache.is_enabled()
    cache.set_enabled(False)
    try:
        index = pd.date_range("2024-01-01", periods=5, freq="M")
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005], index=index)
        result = rolling_information_ratio(returns, benchmark=0.0, window=3)

        assert list(result.index) == list(index)
        assert result.name == "rolling_ir"
        expected = (returns.iloc[-3:] - 0.0).mean() / (returns.iloc[-3:] - 0.0).std(
            ddof=1
        )
        assert result.iloc[-1] == pytest.approx(expected)
    finally:
        cache.set_enabled(original)
