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

@register_metric("volatility")
def volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 12,
) -> float | pd.Series | np.nan:
    """Annualised σ of periodic returns."""

    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("volatility expects a pandas Series or DataFrame")

    if returns.empty:
        return np.nan if isinstance(returns, pd.Series) \
                      else pd.Series(np.nan, index=returns.columns, name="volatility")

    # population std (ddof=0) then annualise
    sigma = returns.std(ddof=0) * np.sqrt(periods_per_year)
    return float(sigma) if isinstance(returns, pd.Series) else sigma.astype(float)

@register_metric("sharpe_ratio")
def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 12,
    risk_free: float = 0.0,           # annual risk‑free rate
) -> float | pd.Series | np.nan:
    """Annualised Sharpe ratio (excess return ÷ σ)."""

    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("sharpe_ratio expects a pandas Series or DataFrame")

    if returns.empty:
        return np.nan if isinstance(returns, pd.Series) \
                      else pd.Series(np.nan, index=returns.columns, name="sharpe_ratio")

    # convert annual rf to per‑period
    rf_per_period = risk_free / periods_per_year
    excess = returns - rf_per_period

    mean_excess = excess.mean()
    sigma = excess.std(ddof=0)

    sharpe = mean_excess / sigma * np.sqrt(periods_per_year)
    return float(sharpe) if isinstance(returns, pd.Series) else sharpe.astype(float)

@register_metric("sortino_ratio")
def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = 12,
    target: float = 0.0,        # minimum acceptable return per *period*
) -> float | pd.Series | np.nan:
    """
    Annualised Sortino ratio:  (mean excess) / (downside σ) * √T
    Downside deviation uses negative returns relative to `target`.
    """

    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("sortino_ratio expects a pandas Series or DataFrame")

    if returns.empty:
        if isinstance(returns, pd.Series):
            return np.nan
        return pd.Series(np.nan, index=returns.columns, name="sortino_ratio")

    # 1. Excess returns relative to target
    excess = returns - target

    # 2. Downside part only  (negative excess → keep, positive → 0)
    downside = excess.clip(upper=0)

    # 3. Mean excess & downside std
    mean_excess = excess.mean()
    downside_std = np.sqrt((downside ** 2).mean())

    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    return float(sortino) if isinstance(returns, pd.Series) else sortino.astype(float)

@register_metric("max_drawdown")
def max_drawdown(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series | np.nan:
    """Maximum drawdown (most negative peak‑to‑trough return)."""

    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("max_drawdown expects a pandas Series or DataFrame")

    if prices.empty:
        return np.nan if isinstance(prices, pd.Series) \
                      else pd.Series(np.nan, index=prices.columns, name="max_drawdown")

    running_max = prices.ffill().cummax()
    draw = (prices / running_max) - 1.0
    mdd = draw.min()                     # Series or scalar, depending on input
    return float(mdd) if isinstance(prices, pd.Series) else mdd.astype(float)

@register_metric("info_ratio")
def information_ratio(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | None = None,
    periods_per_year: int = 12,
) -> float | pd.Series | np.nan:
    """
    Information ratio = annualised active return / tracking error.
    If `benchmark` is None and `returns` is a DataFrame,
    the column‑wise mean is used as the benchmark.
    """

    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("info_ratio expects a pandas Series or DataFrame")

    if returns.empty:
        if isinstance(returns, pd.Series):
            return np.nan
        return pd.Series(np.nan, index=returns.columns, name="info_ratio")

    # --- Align benchmark --------------------------------------------------
    if benchmark is None:
        # DataFrame → peer‑group cross‑sectional mean; Series → zeros
        benchmark = returns.mean(axis=1) if isinstance(returns, pd.DataFrame) else 0.0

    # Reindex to match returns exactly
    benchmark = benchmark.reindex_like(returns)

    # --- Active return & tracking error ----------------------------------
    active = returns - benchmark
    ann_active_ret = active.mean() * periods_per_year
    tracking_err  = active.std(ddof=0) * np.sqrt(periods_per_year)

    info = ann_active_ret / tracking_err
    return float(info) if isinstance(returns, pd.Series) else info.astype(float)
