import numpy as np
import pandas as pd
import pytest

import trend_analysis.metrics as M
from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH

_rng = np.random.default_rng(42)
_periods = 36
_cols = list("ABCD")


def _dummy_returns():
    # Use a fixed seed to ensure reproducible data for each test case
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-31", periods=_periods, freq="ME")
    return pd.DataFrame(
        rng.normal(0.01, 0.02, size=(_periods, len(_cols))), index=dates, columns=_cols
    )


def _dummy_prices():
    # start at 100 and walk
    rets = _dummy_returns()
    return 100 * (1 + rets).cumprod()


# Special function for info_ratio that ensures benchmark and data consistency
def _dummy_returns_with_benchmark():
    """Generate returns data and return both the data and its benchmark (mean
    across columns)."""
    data = _dummy_returns()
    return data, data.mean(axis=1)


# (metric_name, data_fn, kwargs)
CASES = [
    ("volatility", _dummy_returns, {}),
    ("sharpe_ratio", _dummy_returns, {"risk_free": 0.0}),
    ("max_drawdown", _dummy_prices, {}),
    ("sortino_ratio", _dummy_returns, {"target": 0.0}),
    ("info_ratio", _dummy_returns_with_benchmark, {}),
]


@pytest.mark.parametrize("name, data_fn, kw", CASES)
def test_vectorised_metric_matches_explicit_scalar_contract(name, data_fn, kw):
    # The vector path must agree with applying the canonical scalar API per column.
    if name == "info_ratio":
        data, benchmark = data_fn()
        kw = {"benchmark": benchmark}
    else:
        data = data_fn()

    vec_fn = getattr(M, name)
    new_series = vec_fn(data, **kw)

    if name == "sharpe_ratio":
        old_series = pd.Series({c: vec_fn(data[c], risk_free=0.0) for c in data.columns})
    elif name == "sortino_ratio":
        old_series = pd.Series({c: vec_fn(data[c], target=0.0) for c in data.columns})
    else:
        old_series = pd.Series({c: vec_fn(data[c], **kw) for c in data.columns})

    pd.testing.assert_series_equal(
        new_series,
        old_series,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
    )

    one_col = data[_cols[0]]
    new_scalar = vec_fn(one_col, **kw)
    if name == "sharpe_ratio":
        old_scalar = vec_fn(one_col, risk_free=0.0)
    elif name == "sortino_ratio":
        old_scalar = vec_fn(one_col, target=0.0)
    else:
        old_scalar = vec_fn(one_col, **kw)
    assert np.isclose(
        new_scalar,
        old_scalar,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
        equal_nan=True,
    )
