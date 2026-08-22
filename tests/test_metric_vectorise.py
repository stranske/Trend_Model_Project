import numpy as np
import pandas as pd

import trend_analysis.metrics as M
from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH


def _dummy_returns():
    rng = pd.date_range("2020-01-31", periods=24, freq="ME")
    # 24 months × 4 funds of small returns
    data = np.random.default_rng(0).normal(0.01, 0.02, size=(24, 4))
    return pd.DataFrame(data, index=rng, columns=list("ABCD"))


def test_annual_return_vectorised_matches_geometric_golden_contract():
    returns = _dummy_returns()
    new = M.annual_return(returns)
    old = (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0
    pd.testing.assert_series_equal(
        new, old, rtol=NUMERICAL_TOLERANCE_HIGH, atol=NUMERICAL_TOLERANCE_HIGH
    )
