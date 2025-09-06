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


# (metric_name, data_fn, kwargs_factory)
CASES = [
    ("volatility", _dummy_returns, lambda: {}),
    ("sharpe_ratio", _dummy_returns, lambda: {"risk_free": 0.0}),
    ("max_drawdown", _dummy_prices, lambda: {}),
    ("sortino_ratio", _dummy_returns, lambda: {"target": 0.0}),
    (
        "info_ratio",
        _dummy_returns,
        lambda: {"benchmark": _dummy_returns().mean(axis=1)},
    ),
]


@pytest.mark.parametrize("name, data_fn, kw_fn", CASES)
def test_vectorised_metric_matches_legacy(name, data_fn, kw_fn):
    data = data_fn()
    vec_fn = getattr(M, name)
    leg_fn = getattr(L, name)

    kw = kw_fn()
    new_series = vec_fn(data, **kw)
    old_series = leg_fn(data, **kw)

    pd.testing.assert_series_equal(
        new_series,
        old_series,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
    )

    # also test Series input → scalar
    one_col = data[_cols[0]]
    new_scalar = vec_fn(one_col, **kw)
    old_scalar = leg_fn(one_col, **kw)
    assert np.isclose(
        new_scalar,
        old_scalar,
        rtol=NUMERICAL_TOLERANCE_HIGH,
        atol=NUMERICAL_TOLERANCE_HIGH,
        equal_nan=True,
    )
