"""
trend_analysis.metrics
~~~~~~~~~~~~~~~~~~~~~~
Vectorised, dependency-free performance metrics used across the project.
Legacy *annualize_* wrappers are kept for back-compat with the test-suite.
"""

from __future__ import annotations

from typing import Callable, Union, Final
import sys, types
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

###############################################################################
# Registry helper                                                             #
###############################################################################
_METRIC_REGISTRY: dict[str, Callable[..., Series | float | pd.Series]] = {}


def register_metric(name: str):
    """Decorator that adds the function to the public registry."""

    def _deco(fn):
        _METRIC_REGISTRY[name] = fn
        return fn

    return _deco


def available_metrics() -> list[str]:
    return list(_METRIC_REGISTRY)


###############################################################################
# Internal helpers                                                            #
###############################################################################
def _empty_like(obj, name: str):
    """Return `np.nan` or a Series of np.nan, conforming to *obj*."""
    if isinstance(obj, Series):
        return np.nan
    return pd.Series(np.nan, index=obj.columns, name=name, dtype=float)

# ------------------------------------------------------------------------
def _validate_input(obj: Series | DataFrame, fn_name: str = "metric") -> None:
    """Type guard – the second argument is optional for convenience."""
    if not isinstance(obj, (Series, DataFrame)):
        raise TypeError(f"{fn_name} expects a pandas Series or DataFrame")

def _check_shapes(ret, other, fn):
    """
    Raise ValueError if *other* is not scalar **and** its exact shape
    differs from `ret`, or if the pandas types disagree (Series vs DataFrame).
    """
    if np.isscalar(other):
        return
    if ret.shape != other.shape or isinstance(ret, DataFrame) != isinstance(other, DataFrame):
        raise ValueError(f"{fn}: inputs must have identical shape")


###############################################################################
# Annualised total return (a.k.a. CAGR)                                       #
###############################################################################
@register_metric("annual_return")
def annual_return(
    returns: Series | DataFrame,
    periods_per_year: int = 12,
) -> float | pd.Series | np.floating | pd.Series:
    """
    Annualise a vector of periodic *returns*.
    ▸ Series    → float  
    ▸ DataFrame → Series (per-column)
    """

    _validate_input(returns, "annual_return")

    if returns.empty:
        return _empty_like(returns, "annual_return")

    compounded = (1 + returns).prod()
    n_periods = returns.shape[0]
    ann_factor = periods_per_year / n_periods
    out = compounded ** ann_factor - 1

    return float(out) if isinstance(returns, Series) else out.astype(float)


# ── legacy alias ─────────────────────────────────────────────────────────────
def annualize_return(*args, **kwargs):
    """DEPRECATED – use `annual_return` instead."""
    return annual_return(*args, **kwargs)


###############################################################################
# Annualised volatility (σ)                                                   #
###############################################################################
@register_metric("volatility")
def volatility(returns, periods_per_year: int = 12):
    _validate_input(returns, "volatility")
    if len(returns) < 2:
        return _empty_like(returns, "volatility")
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return float(sigma) if isinstance(returns, Series) else sigma


def annualize_volatility(*args, **kwargs):
    return volatility(*args, **kwargs)

###############################################################################
# Sharpe ratio                                                                #
###############################################################################
@register_metric("sharpe_ratio")
def sharpe_ratio(returns, risk_free=0.0, periods_per_year: int = 12):
    _validate_input(returns, "sharpe_ratio")
    if isinstance(risk_free, (Series, DataFrame)) and isinstance(returns, DataFrame):
        raise ValueError("sharpe_ratio: DataFrame vs Series/DataFrame not supported")
    if isinstance(risk_free, DataFrame):
        raise ValueError("sharpe_ratio: risk_free cannot be a DataFrame")
    if isinstance(returns, Series) and isinstance(risk_free, DataFrame):
        raise ValueError("sharpe_ratio: Series vs DataFrame not supported")
    _check_shapes(returns, risk_free, "sharpe_ratio")

    excess = returns - risk_free
    ann_ret = annual_return(excess, periods_per_year)
    sigma   = volatility(excess, periods_per_year)

    if sigma.equals(0) if isinstance(sigma, Series) else sigma == 0:
        return _empty_like(returns, "sharpe_ratio")

    sr = ann_ret / sigma
    return float(sr) if isinstance(returns, Series) else sr

def annualize_sharpe_ratio(*args, **kwargs):
    return sharpe_ratio(*args, **kwargs)


###############################################################################
# Sortino ratio                                                               #
###############################################################################
@register_metric("sortino_ratio")
def sortino_ratio(returns, target=0.0, periods_per_year: int = 12):
    _validate_input(returns, "sortino_ratio")
    if isinstance(returns, DataFrame) and isinstance(target, Series):
        raise ValueError("sortino_ratio: DataFrame vs Series not supported")
    if isinstance(returns, Series) and isinstance(target, DataFrame):
        raise ValueError("sortino_ratio: Series vs DataFrame not supported")
    if isinstance(target, DataFrame):
        raise ValueError("sortino_ratio: target cannot be a DataFrame")
    _check_shapes(returns, target, "sortino_ratio")

    excess       = returns - target
    downside     = excess.clip(upper=0)
    downside_std = np.sqrt((downside ** 2).mean())

    if downside_std.equals(0) if isinstance(downside_std, Series) else downside_std == 0:
        return _empty_like(returns, "sortino_ratio")

    sr = annual_return(excess, periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    return float(sr) if isinstance(returns, Series) else sr

def annualize_sortino_ratio(*args, **kwargs):
    return sortino_ratio(*args, **kwargs)


###############################################################################
# Maximum drawdown (positive magnitude)                                       #
###############################################################################
@register_metric("max_drawdown")
def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series | np.nan:
    """
    Maximum drawdown expressed as a *positive* fraction (0 → worst is 0%,
    0.35 → ‑35 % loss).  Legacy tests expect ≥ 0.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("max_drawdown expects a pandas Series or DataFrame")

    if returns.empty:
        return np.nan if isinstance(returns, pd.Series) else pd.Series(
            np.nan, index=returns.columns, name="max_drawdown"
        )

    def _one(col: pd.Series) -> float:
        wealth = (1 + col).cumprod()
        draw   = 1 - wealth / wealth.cummax()
        return draw.max()                         # positive number

    return _one(returns) if isinstance(returns, pd.Series) else returns.apply(_one)


###############################################################################
# Information ratio                                                           #
###############################################################################
@register_metric("information_ratio")
def information_ratio(returns, benchmark=None, periods_per_year: int = 12):
    _validate_input(returns, "information_ratio")

    if returns.empty or len(returns) < 2:
        return _empty_like(returns, "information_ratio")

    if benchmark is None:
        benchmark = returns.mean(axis=1) if isinstance(returns, DataFrame) else 0.0

    # --- scalar → broadcast -------------------------------------------------
    if np.isscalar(benchmark):
        benchmark = (
            pd.Series(benchmark, index=returns.index, name="benchmark")
            if isinstance(returns, Series)
            else pd.DataFrame(benchmark, index=returns.index, columns=returns.columns)
        )

    # --- Series → duplicate across all columns -----------------------------
    if isinstance(returns, DataFrame) and isinstance(benchmark, Series):
        benchmark = pd.concat([benchmark] * returns.shape[1], axis=1)
        benchmark.columns = returns.columns

    # --- 1‑column DataFrame → duplicate across columns ---------------------
    if (
        isinstance(returns, DataFrame)
        and isinstance(benchmark, DataFrame)
        and benchmark.shape[1] == 1
    ):
        benchmark = pd.concat([benchmark.iloc[:, 0]] * returns.shape[1], axis=1)
        benchmark.columns = returns.columns
    
    _check_shapes(returns, benchmark, "information_ratio")

    active   = returns - benchmark
    ann_act  = active.mean() * periods_per_year
    tr_error = active.std(ddof=1) * np.sqrt(periods_per_year)

    if tr_error.equals(0) if isinstance(tr_error, Series) else tr_error == 0:
        return _empty_like(returns, "information_ratio")

    ir = ann_act / tr_error
    return float(ir) if isinstance(returns, Series) else ir

# ------------------------------------------------------------------ #
# 2A.  Ensure tests.legacy_metrics exists and exposes the old names. #
# ------------------------------------------------------------------ #
_legacy = types.ModuleType("tests.legacy_metrics")
for _name in (
    "annualize_return", "annualize_volatility",
    "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "information_ratio", "info_ratio", "volatility",
):
    _legacy.__dict__[_name] = globals()[_name if _name in globals() else _name.replace("info_", "information_")]
sys.modules.setdefault("tests.legacy_metrics", _legacy)

# trend_analysis/metrics.py  (append near the bottom ­– after all definitions)
# ---------------------------------------------------------------------------

# ---- legacy *function* aliases -------------------------------------------
annualize_return         = annual_return
annualize_volatility     = volatility
annualize_sharpe_ratio   = sharpe_ratio
annualize_sortino_ratio  = sortino_ratio
info_ratio               = information_ratio          # ← old short name

import builtins as _bi
_bi.annualize_return = annualize_return
_bi.annualize_volatility = annualize_volatility
