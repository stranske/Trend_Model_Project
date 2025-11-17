import numpy as np
import pandas as pd

from trend_analysis.signals import TrendSpec, compute_trend_signals


def _sample_returns() -> pd.DataFrame:
    index = pd.RangeIndex(8)
    data = {
        "asset_a": np.linspace(-0.02, 0.03, len(index)),
        "asset_b": np.linspace(0.01, -0.015, len(index)),
    }
    return pd.DataFrame(data, index=index)


def test_compute_trend_signals_is_causal_without_vol_adjust() -> None:
    returns = _sample_returns()
    spec = TrendSpec(window=3, min_periods=3)

    baseline = compute_trend_signals(returns, spec)
    tweaked = returns.copy()
    tweaked.iloc[-1] = 999.0
    shifted = compute_trend_signals(tweaked, spec)

    pd.testing.assert_frame_equal(baseline, shifted)


def test_compute_trend_signals_is_causal_with_vol_adjust() -> None:
    returns = _sample_returns()
    spec = TrendSpec(window=4, min_periods=4, vol_adjust=True, vol_target=0.5)

    baseline = compute_trend_signals(returns, spec)
    tweaked = returns.copy()
    tweaked.iloc[-1] = -999.0
    shifted = compute_trend_signals(tweaked, spec)

    pd.testing.assert_frame_equal(baseline, shifted)
