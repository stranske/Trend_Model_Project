"""Core performance metrics used across the project."""

# mypy: ignore-errors

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from trend_analysis.metrics import information_ratio


def _validate_input(obj: Series | DataFrame) -> None:
    if not isinstance(obj, (Series, DataFrame)):
        raise TypeError("Input must be pandas Series or DataFrame")


def _apply(  # helper to handle Series/DataFrame uniformly
    obj: Series | DataFrame, func, axis: int
) -> Series | float:
    if isinstance(obj, Series):
        return func(obj.dropna())
    return obj.apply(lambda col: func(col.dropna()), axis=axis)


def annual_return(
    returns: Series | DataFrame, periods_per_year: int = 12, axis: int = 0
) -> Series | float:
    """Geometric annualised return.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Periodic returns in decimal form.
    periods_per_year : int, default 12
        Number of compounding periods per year.
    axis : int, default 0
        Axis along which to calculate for DataFrames.

    Returns
    -------
    pandas.Series or float
        Annualised return(s).

    Example
    -------
    >>> import pandas as pd
    >>> s = pd.Series([0.02, -0.01, 0.03])
    >>> annual_return(s)
    0.236090...  # doctest: +ELLIPSIS
    """
    _validate_input(returns)

    def _calc(x: Series) -> float:
        if x.empty:
            return np.nan
        growth = (1 + x).prod()
        n = len(x)
        if growth <= 0:
            return -1.0
        return growth ** (periods_per_year / n) - 1

    return _apply(returns, _calc, axis)


def volatility(
    returns: Series | DataFrame, periods_per_year: int = 12, axis: int = 0
) -> Series | float:
    """Annualised volatility of returns."""
    _validate_input(returns)

    def _calc(x: Series) -> float:
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) * np.sqrt(periods_per_year)

    return _apply(returns, _calc, axis)


def sharpe_ratio(
    returns: Series | DataFrame,
    rf: Series | DataFrame,
    periods_per_year: int = 12,
    axis: int = 0,
) -> Series | float:
    """Annualised Sharpe ratio."""
    _validate_input(returns)
    _validate_input(rf)

    def _calc(r: Series, rf_s: Series) -> float:
        df = pd.DataFrame({"r": r, "rf": rf_s}).dropna()
        if len(df) < 2:
            return np.nan
        excess = df["r"] - df["rf"]
        ann_excess_ret = annual_return(excess, periods_per_year)
        ann_excess_vol = volatility(excess, periods_per_year)
        if ann_excess_vol == 0 or np.isnan(ann_excess_vol):
            return np.nan
        return ann_excess_ret / ann_excess_vol

    if isinstance(returns, Series) and isinstance(rf, Series):
        return _calc(returns, rf)

    if isinstance(returns, DataFrame) and isinstance(rf, Series):
        rf = DataFrame({c: rf for c in returns.columns})

    if isinstance(returns, Series) and isinstance(rf, DataFrame):
        returns = DataFrame({c: returns for c in rf.columns})

    if isinstance(returns, DataFrame) and isinstance(rf, DataFrame):
        return DataFrame({col: _calc(returns[col], rf[col]) for col in returns.columns}).squeeze(
            axis=1
        )

    raise TypeError("returns and rf must be Series or DataFrame of compatible shape")


def sortino_ratio(
    returns: Series | DataFrame,
    rf: Series | DataFrame,
    periods_per_year: int = 12,
    axis: int = 0,
) -> Series | float:
    """Annualised Sortino ratio."""
    _validate_input(returns)
    _validate_input(rf)

    def _calc(r: Series, rf_s: Series) -> float:
        df = pd.DataFrame({"r": r, "rf": rf_s}).dropna()
        if len(df) < 2:
            return np.nan
        excess = df["r"] - df["rf"]
        growth = (1 + excess).prod()
        ann_ret = growth ** (periods_per_year / len(excess)) - 1 if growth > 0 else np.nan
        downside = excess[excess < 0]
        if downside.empty:
            return np.nan
        down_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
        if down_vol == 0 or np.isnan(down_vol):
            return np.nan
        return ann_ret / down_vol

    if isinstance(returns, Series) and isinstance(rf, Series):
        return _calc(returns, rf)

    if isinstance(returns, DataFrame) and isinstance(rf, Series):
        rf = DataFrame({c: rf for c in returns.columns})

    if isinstance(returns, Series) and isinstance(rf, DataFrame):
        returns = DataFrame({c: returns for c in rf.columns})

    if isinstance(returns, DataFrame) and isinstance(rf, DataFrame):
        return DataFrame({col: _calc(returns[col], rf[col]) for col in returns.columns}).squeeze(
            axis=1
        )

    raise TypeError("returns and rf must be Series or DataFrame of compatible shape")


def max_drawdown(returns: Series | DataFrame, axis: int = 0) -> Series | float:
    """Maximum drawdown of cumulative returns."""
    _validate_input(returns)

    def _calc(x: Series) -> float:
        if x.empty:
            return np.nan
        wealth = (1 + x).cumprod()
        dd = 1 - wealth / wealth.cummax()
        return dd.max()

    return _apply(returns, _calc, axis)


annualize_return = annual_return
information_ratio = info_ratio = information_ratio
