import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_analysis import metrics


def test_annualize_return_series():
    s = pd.Series([0.02, -0.01, 0.03])
    result = metrics.annualize_return(s)
    expected = (1 + s).prod() ** (12 / len(s)) - 1
    assert np.isclose(result, expected)


def test_annualize_return_empty():
    assert np.isnan(metrics.annualize_return(pd.Series(dtype=float)))


def test_annualize_return_negative_growth():
    s = pd.Series([-1.0])
    assert metrics.annualize_return(s) == -1.0


def test_annualize_volatility_dataframe():
    df = pd.DataFrame({"a": [0.01, 0.02, 0.03, -0.01]})
    res = metrics.annualize_volatility(df)
    expected = df.std(ddof=1) * np.sqrt(12)
    assert np.allclose(res.values, expected.values)


def test_annualize_volatility_single():
    s = pd.Series([0.01])
    assert np.isnan(metrics.annualize_volatility(s))


def test_sharpe_ratio_dataframe_series():
    r = pd.DataFrame({"a": [0.02, 0.03, -0.01]})
    rf = pd.Series([0.01, 0.01, 0.01])
    res = metrics.sharpe_ratio(r, rf)
    assert isinstance(res, pd.Series)


def test_sharpe_ratio_zero_vol():
    r = pd.Series([0.01, 0.01])
    rf = pd.Series([0.0, 0.0])
    assert np.isnan(metrics.sharpe_ratio(r, rf))


def test_sharpe_ratio_single_period():
    r = pd.Series([0.05])
    rf = pd.Series([0.01])
    assert np.isnan(metrics.sharpe_ratio(r, rf))


def test_sharpe_ratio_type_error():
    with pytest.raises(TypeError):
        metrics.sharpe_ratio([0.1], [0.1])


def test_sortino_ratio_dataframe():
    r = pd.DataFrame({"a": [0.02, -0.03, 0.01]})
    r = pd.DataFrame({"a": [0.02, -0.03, -0.04]})
    rf = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    res = metrics.sortino_ratio(r, rf)
    assert isinstance(res, pd.Series)


def test_sortino_ratio_no_downside():
    r = pd.Series([0.05, 0.04])
    rf = pd.Series([0.01, 0.01])
    assert np.isnan(metrics.sortino_ratio(r, rf))


def test_sortino_ratio_single_period():
    r = pd.Series([-0.02])
    rf = pd.Series([0.0])
    assert np.isnan(metrics.sortino_ratio(r, rf))


def test_sortino_ratio_series_dataframe():
    r = pd.Series([-0.01, 0.02])
    rf = pd.DataFrame({"a": [0.0, 0.0]})
    res = metrics.sortino_ratio(r, rf)
    assert isinstance(res, pd.Series)


def test_sortino_ratio_bad_types():
    with pytest.raises(TypeError):
        metrics.sortino_ratio([0.1], [0.1])


def test_max_drawdown_dataframe():
    df = pd.DataFrame({"a": [0.1, -0.05, 0.02, -0.02]})
    result = metrics.max_drawdown(df)
    wealth = (1 + df["a"]).cumprod()
    expected = (1 - wealth / wealth.cummax()).max()
    assert np.isclose(result["a"], expected)


def test_max_drawdown_empty():
    s = pd.Series(dtype=float)
    assert np.isnan(metrics.max_drawdown(s))
