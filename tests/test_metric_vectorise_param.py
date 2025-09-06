import numpy as np
import pandas as pd
import pytest

import tests.legacy_metrics as L
import trend_analysis.metrics as M
from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH

_rng = np.random.default_rng(42)
_periods = 36
_cols = list("ABCD")


def _dummy_returns():
    dates = pd.date_range("2021-01-31", periods=_periods, freq="ME")
    return pd.DataFrame(
        _rng.normal(0.01, 0.02, size=(_periods, len(_cols))), index=dates, columns=_cols
    )


def _dummy_prices():
    # start at 100 and walk
    rets = _dummy_returns()
    return 100 * (1 + rets).cumprod()


CASES = [
    ("volatility", _dummy_returns, lambda data: {}),
    ("sharpe_ratio", _dummy_returns, lambda data: {"risk_free": 0.0}),
    ("max_drawdown", _dummy_prices, lambda data: {}),
    ("info_ratio", _dummy_returns, lambda data: {"benchmark": data.mean(axis=1)}),
]


@pytest.mark.parametrize("name, data_fn, kw_factory", CASES)
def test_vectorised_metric_matches_legacy(name, data_fn, kw_factory):
    data = data_fn()
    vec_fn = getattr(M, name)
    leg_fn = getattr(L, name)

    kw = kw_factory(data)
    new_series = vec_fn(data, **kw)

    if name == "sharpe_ratio":
        rf = pd.Series(0.0, index=data.index)
        old_series = pd.Series({c: leg_fn(data[c], rf) for c in data.columns})
    else:
        old_series = leg_fn(data, **kw)

    pd.testing.assert_series_equal(
        new_series,
        old_series,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
    )

    one_col = data[_cols[0]]
    kw = kw_factory(one_col.to_frame())
    new_scalar = vec_fn(one_col, **kw)
    if name == "sharpe_ratio":
        rf = pd.Series(0.0, index=one_col.index)
        old_scalar = leg_fn(one_col, rf)
    else:
        old_scalar = leg_fn(one_col, **kw)
    assert np.isclose(
        new_scalar,
        old_scalar,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
        equal_nan=True,
    )
