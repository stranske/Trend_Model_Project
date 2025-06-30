import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_analysis import metrics

import pandas as pd
import numpy as np


def test_annualize_return_series():
    s = pd.Series([0.02, -0.01, 0.03])
    result = metrics.annualize_return(s)
    expected = (1 + s).prod() ** (12 / len(s)) - 1
    assert np.isclose(result, expected)


def test_annualize_volatility_dataframe():
    df = pd.DataFrame({"a": [0.01, 0.02, 0.03, -0.01]})
    res = metrics.annualize_volatility(df)
    expected = df.std(ddof=1) * np.sqrt(12)
    assert np.allclose(res.values, expected.values)


def test_sharpe_ratio_simple():
    r = pd.Series([0.02, 0.03, -0.01])
    rf = pd.Series([0.01, 0.01, 0.01])
    res = metrics.sharpe_ratio(r, rf)
    ex = r - rf
    expected = metrics.annualize_return(ex) / metrics.annualize_volatility(ex)
    assert np.isclose(res, expected, equal_nan=True)


def test_sortino_ratio_no_downside():
    r = pd.Series([0.05, 0.04])
    rf = pd.Series([0.01, 0.01])
    assert np.isnan(metrics.sortino_ratio(r, rf))


def test_max_drawdown_dataframe():
    df = pd.DataFrame({"a": [0.1, -0.05, 0.02, -0.02]})
    result = metrics.max_drawdown(df)
    wealth = (1 + df["a"]).cumprod()
    expected = (1 - wealth / wealth.cummax()).max()
    assert np.isclose(result["a"], expected)
