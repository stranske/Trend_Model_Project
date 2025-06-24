import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_analysis import metrics

import pandas as pd
import numpy as np
import pytest


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

def test_validate_input_error():
    with pytest.raises(TypeError):
        metrics._validate_input([1, 2, 3])


def test_annualize_return_edges():
    assert np.isnan(metrics.annualize_return(pd.Series([], dtype=float)))
    s = pd.Series([-2.0])
    assert metrics.annualize_return(s) == -1.0
    df = pd.DataFrame({"a": [0.01, 0.02], "b": [0.0, 0.0]})
    res = metrics.annualize_return(df, axis=0)
    assert isinstance(res, pd.Series)


def test_annualize_volatility_short_series():
    s = pd.Series([0.01])
    assert np.isnan(metrics.annualize_volatility(s))
    df = pd.DataFrame({"a": [0.01, 0.02]})
    assert isinstance(metrics.annualize_volatility(df), pd.Series)


def test_sharpe_ratio_branches(monkeypatch):
    r = pd.DataFrame({"a": [0.02, 0.03, -0.01], "b": [0.01, 0.01, 0.01]})
    rf = pd.Series([0.0, 0.0, 0.0])

    class WrappedDF(pd.DataFrame):
        def __init__(self, data=None, **kwargs):
            if isinstance(data, dict):
                data = {k: (v if hasattr(v, "__len__") and not isinstance(v, float) else [v]) for k, v in data.items()}
            super().__init__(data, **kwargs)

    monkeypatch.setattr(metrics, "DataFrame", WrappedDF)
    r = WrappedDF(r)

    res = metrics.sharpe_ratio(r, rf)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    res2 = metrics.sharpe_ratio(r["a"], WrappedDF(WrappedDF(WrappedDF(rf.to_frame("x")))))
    assert isinstance(res2, pd.Series)
    assert np.isnan(metrics.sharpe_ratio(pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0])))
    with pytest.raises(TypeError):
        metrics.sharpe_ratio([1, 2], rf)


def test_sortino_ratio_branches(monkeypatch):
    r = pd.DataFrame({"a": [0.02, -0.03, 0.01], "b": [-0.01, -0.02, -0.01]})
    rf = pd.Series([0.0, 0.0, 0.0])

    class WrappedDF(pd.DataFrame):
        def __init__(self, data=None, **kwargs):
            if isinstance(data, dict):
                data = {k: (v if hasattr(v, "__len__") and not isinstance(v, float) else [v]) for k, v in data.items()}
            super().__init__(data, **kwargs)

    monkeypatch.setattr(metrics, "DataFrame", WrappedDF)
    r = WrappedDF(r)

    assert isinstance(metrics.sortino_ratio(r, rf), (pd.Series, pd.DataFrame))
    res = metrics.sortino_ratio(r["a"], WrappedDF(rf.to_frame("x")))
    assert isinstance(res, pd.Series)
    assert np.isnan(metrics.sortino_ratio(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02])))
    with pytest.raises(TypeError):
        metrics.sortino_ratio([0.1], rf)


def test_max_drawdown_empty_series():
    assert np.isnan(metrics.max_drawdown(pd.Series(dtype=float)))


def test_sharpe_ratio_short_series():
    r = pd.Series([0.01])
    rf = pd.Series([0.0])
    assert np.isnan(metrics.sharpe_ratio(r, rf))


def test_sharpe_ratio_typeerror_branch(monkeypatch):
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    class Dummy:
        pass
    with pytest.raises(TypeError):
        metrics.sharpe_ratio(Dummy([0.1]), Dummy([0.2]))


def test_sortino_ratio_short_series():
    r = pd.Series([0.02])
    rf = pd.Series([0.0])
    assert np.isnan(metrics.sortino_ratio(r, rf))


def test_sortino_ratio_typeerror_branch(monkeypatch):
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    class Dummy:
        pass
    with pytest.raises(TypeError):
        metrics.sortino_ratio(Dummy([0.1]), Dummy([0.2]))


def test_sharpe_ratio_dataframe_false_branch(monkeypatch):
    class MyDF(pd.DataFrame):
        pass
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    monkeypatch.setattr(metrics, "DataFrame", MyDF)
    with pytest.raises(TypeError):
        metrics.sharpe_ratio(pd.DataFrame({"a": [0.1]}), MyDF({"a": [0.1]}))


def test_sortino_ratio_dataframe_false_branch(monkeypatch):
    class MyDF(pd.DataFrame):
        pass
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    monkeypatch.setattr(metrics, "DataFrame", MyDF)
    with pytest.raises(TypeError):
        metrics.sortino_ratio(pd.DataFrame({"a": [0.1]}), MyDF({"a": [0.1]}))

