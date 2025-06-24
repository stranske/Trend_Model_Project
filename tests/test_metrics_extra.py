import numpy as np
import pandas as pd
import pytest

from trend_analysis import metrics


def test_validate_input():
    with pytest.raises(TypeError):
        metrics.annualize_return([1, 2])  # type: ignore


def test_annualize_return_branches():
    s = pd.Series([], dtype=float)
    assert np.isnan(metrics.annualize_return(s))
    neg = pd.Series([-1.0])
    assert metrics.annualize_return(neg) == -1.0
    df = pd.DataFrame({"a": [0.01, 0.02]})
    res = metrics.annualize_return(df)
    assert isinstance(res, pd.Series)


def test_annualize_volatility_branches():
    s = pd.Series([0.1])
    assert np.isnan(metrics.annualize_volatility(s))
    df = pd.DataFrame({"a": [0.1, 0.2, 0.3]})
    res = metrics.annualize_volatility(df)
    assert isinstance(res, pd.Series)


def test_sharpe_ratio_variants(monkeypatch):
    r = pd.Series([0.02, 0.01, 0.03])
    rf = pd.Series([0.01, 0.01, 0.01])
    assert metrics.sharpe_ratio(r, rf) > 0
    df = pd.DataFrame({"a": r, "b": r})
    # DataFrame + Series path triggers ValueError under pandas when results
    # are scalars. Exercise the branch accordingly.
    with pytest.raises(ValueError):
        metrics.sharpe_ratio(df, rf)
    # Series + DataFrame path behaves the same
    with pytest.raises(ValueError):
        metrics.sharpe_ratio(r, df)
    # Both DataFrames likewise raise ValueError
    with pytest.raises(ValueError):
        metrics.sharpe_ratio(df, df)
    zero = pd.Series([0.0, 0.0])
    assert np.isnan(metrics.sharpe_ratio(zero, zero))
    with pytest.raises(TypeError):
        metrics.sharpe_ratio([0.1], [0.1])  # type: ignore

    # Patch input validation to hit the final TypeError branch
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    with pytest.raises(TypeError):
        metrics.sharpe_ratio(pd.Series([1]), (1,))


def test_sharpe_ratio_short_series():
    r = pd.Series([0.1])
    rf = pd.Series([0.0])
    assert np.isnan(metrics.sharpe_ratio(r, rf))


def test_sortino_ratio_variants(monkeypatch):
    r = pd.Series([0.05, 0.04])
    rf = pd.Series([0.01, 0.01])
    assert np.isnan(metrics.sortino_ratio(r, rf))
    df = pd.DataFrame({"a": r, "b": r})
    with pytest.raises(ValueError):
        metrics.sortino_ratio(df, rf)
    with pytest.raises(ValueError):
        metrics.sortino_ratio(r, df)
    with pytest.raises(ValueError):
        metrics.sortino_ratio(df, df)
    with pytest.raises(TypeError):
        metrics.sortino_ratio([0.1], [0.1])  # type: ignore
    monkeypatch.setattr(metrics, "_validate_input", lambda obj: None)
    with pytest.raises(TypeError):
        metrics.sortino_ratio(pd.Series([1]), (1,))


def test_sortino_ratio_short_series():
    r = pd.Series([0.1])
    rf = pd.Series([0.0])
    assert np.isnan(metrics.sortino_ratio(r, rf))


def test_sortino_ratio_compute():
    r = pd.Series([0.1, -0.2, 0.1, -0.1])
    rf = pd.Series([0.0, 0.0, 0.0, 0.0])
    val = metrics.sortino_ratio(r, rf)
    assert not np.isnan(val)


def test_max_drawdown():
    s = pd.Series([0.1, -0.1, 0.05])
    assert metrics.max_drawdown(s) >= 0
    df = pd.DataFrame({"a": s})
    assert isinstance(metrics.max_drawdown(df), pd.Series)
    empty = pd.Series([], dtype=float)
    assert np.isnan(metrics.max_drawdown(empty))



