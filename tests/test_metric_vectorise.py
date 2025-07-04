import numpy as np
import pandas as pd
import trend_analysis.metrics as M
import tests.legacy_metrics as L


def _dummy_returns():
    rng = pd.date_range("2020-01-31", periods=24, freq="ME")
    # 24 months × 4 funds of small returns
    data = np.random.default_rng(0).normal(0.01, 0.02, size=(24, 4))
    return pd.DataFrame(data, index=rng, columns=list("ABCD"))


def test_annual_return_vectorised_equals_legacy():
    prices = _dummy_returns()
    new = M.annualize_return(prices)
    old = L.annualize_return(prices)
    pd.testing.assert_series_equal(new, old, rtol=1e-12, atol=1e-12)
