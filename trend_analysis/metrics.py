"""Core performance metrics used across the project."""

# mypy: ignore-errors

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Callable, Union

# -------------------------------------------------------------------
_METRIC_REGISTRY: dict[str, Callable[..., pd.Series]] = {}


def register_metric(name: str):
    def deco(func):
        _METRIC_REGISTRY[name] = func
        return func

    return deco


def available_metrics() -> list[str]:
    return list(_METRIC_REGISTRY)


def _validate_input(obj: Series | DataFrame) -> None:
    if not isinstance(obj, (Series, DataFrame)):
        raise TypeError("Input must be pandas Series or DataFrame")


def _apply(  # helper to handle Series/DataFrame uniformly
    obj: Series | DataFrame, func, axis: int
) -> Series | float:
    if isinstance(obj, Series):
        return func(obj.dropna())
    return obj.apply(lambda col: func(col.dropna()), axis=axis)


# -------------------------------------------------------------------
# Vectorised annualised total return
# -------------------------------------------------------------------
@register_metric("annual_return")
def annualize_return(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 12,
    axis: int = 0,
) -> Union[float, pd.Series, np.floating, pd.Series]:
    """
    Annualise periodic *returns*.

    ▸ If `returns` is a Series  →  scalar float
    ▸ If `returns` is a DataFrame → Series indexed by column
    Returns `np.nan` for empty input (legacy behaviour).
    """

    # -- 1. type guard (legacy tests expect TypeError) --------------
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("annualize_return expects a pandas Series or DataFrame")

    # -- 2. empty input --------------------------------------------
    if returns.empty:
        if isinstance(returns, pd.Series):
            return np.nan
        return pd.Series(
            np.nan, index=returns.columns, name="annual_return", dtype=float
        )

    # -- 3. compound total return ----------------------------------
    # (1 + r1)·(1 + r2)… − 1   (broadcasts across columns automatically)
    compounded = (1 + returns).prod()

    # -- 4. annualisation factor -----------------------------------
    n_periods = returns.shape[0]
    ann_factor = periods_per_year / n_periods

    # -- 5. annualised return --------------------------------------
    ann_ret = compounded**ann_factor - 1

    # -- 6. preserve legacy output type ----------------------------
    return float(ann_ret) if isinstance(returns, pd.Series) else ann_ret.astype(float)

    def _calc(x: Series) -> float:
        if x.empty:
            return np.nan
        growth = (1 + x).prod()
        n = len(x)
        if growth <= 0:
            return -1.0
        return growth ** (periods_per_year / n) - 1

    return _apply(returns, _calc, axis)


def annualize_volatility(
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
        ann_excess_ret = annualize_return(excess, periods_per_year)
        ann_excess_vol = annualize_volatility(excess, periods_per_year)
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
        return DataFrame(
            {col: _calc(returns[col], rf[col]) for col in returns.columns}
        ).squeeze(axis=1)

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
        ann_ret = (
            growth ** (periods_per_year / len(excess)) - 1 if growth > 0 else np.nan
        )
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
        return DataFrame(
            {col: _calc(returns[col], rf[col]) for col in returns.columns}
        ).squeeze(axis=1)

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
