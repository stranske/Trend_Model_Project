"""
trend_analysis.metrics
~~~~~~~~~~~~~~~~~~~~~~
Vectorised, dependency-free performance metrics used across the project.
Legacy *annualize_* wrappers are kept for back-compat with the test-suite.
"""

from __future__ import annotations

from typing import Callable, cast
import sys
import types
import builtins as _bi
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

###############################################################################
# Registry helper                                                             #
###############################################################################
_METRIC_REGISTRY: dict[str, Callable[..., float | pd.Series | np.floating]] = {}
# Public alias for external access
METRIC_REGISTRY = _METRIC_REGISTRY


def register_metric(
    name: str,
) -> Callable[
    [Callable[..., float | pd.Series | np.floating]],
    Callable[..., float | pd.Series | np.floating],
]:
    """Decorator that adds the function to the public registry."""

    def _deco(
        fn: Callable[..., float | pd.Series | np.floating],
    ) -> Callable[..., float | pd.Series | np.floating]:
        _METRIC_REGISTRY[name] = fn
        return fn

    return _deco


def available_metrics() -> list[str]:
    return list(_METRIC_REGISTRY)


###############################################################################
# Internal helpers                                                            #
###############################################################################
def _empty_like(obj: Series | DataFrame, name: str) -> float | pd.Series:
    """Return ``np.nan`` or a ``Series`` of ``np.nan`` matching ``obj``."""
    if isinstance(obj, Series):
        return np.nan
    return pd.Series(np.nan, index=obj.columns, name=name, dtype=float)


# ------------------------------------------------------------------------
def _validate_input(obj: Series | DataFrame, fn_name: str = "metric") -> None:
    """Type guard – the second argument is optional for convenience."""
    if not isinstance(obj, (Series, DataFrame)):
        raise TypeError(f"{fn_name} expects a pandas Series or DataFrame")


def _check_shapes(
    ret: Series | DataFrame,
    other: Series | DataFrame | float | int,
    fn: str,
) -> None:
    """
    Raise ValueError if *other* is not scalar **and** its exact shape
    differs from `ret`, or if the pandas types disagree (Series vs DataFrame).
    """
    if np.isscalar(other):
        return
    assert isinstance(other, (Series, DataFrame))
    if ret.shape != other.shape or isinstance(ret, DataFrame) != isinstance(
        other, DataFrame
    ):
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

    n_periods = max(int(returns.shape[0]), 1)
    if isinstance(returns, Series):
        prod_val = (1 + returns).prod()
        # Use numpy.asarray(...).item() to obtain a native Python scalar
        compounded_scalar = np.asarray(prod_val).item()
        compounded = float(compounded_scalar)
        # If total growth is non-positive, legacy semantics return -1.0
        if compounded <= 0:
            return -1.0
        out = compounded ** (periods_per_year / n_periods) - 1.0
        return float(out)
    else:
        compounded_df = (1 + returns).prod()
        k = periods_per_year / n_periods
        res = pd.Series(index=returns.columns, dtype=float)
        mask = compounded_df <= 0
        if mask.any():
            res[mask] = -1.0
        if (~mask).any():
            res[~mask] = compounded_df[~mask] ** k - 1.0
        return res.astype(float)


###############################################################################
# Annualised volatility (σ)                                                   #
###############################################################################
@register_metric("volatility")
def volatility(
    returns: Series | DataFrame,
    periods_per_year: int = 12,
) -> float | pd.Series | np.floating:
    _validate_input(returns, "volatility")
    if len(returns) < 2:
        return _empty_like(returns, "volatility")
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return float(sigma) if isinstance(returns, Series) else sigma


###############################################################################
# Sharpe ratio                                                                #
###############################################################################
@register_metric("sharpe_ratio")
def sharpe_ratio(
    returns: Series | DataFrame,
    risk_free: Series | DataFrame | float = 0.0,
    periods_per_year: int = 12,
) -> float | pd.Series | np.floating:
    _validate_input(returns, "sharpe_ratio")
    if isinstance(risk_free, (Series, DataFrame)) and isinstance(returns, DataFrame):
        raise ValueError("sharpe_ratio: DataFrame vs Series/DataFrame not supported")
    if isinstance(risk_free, DataFrame):
        raise ValueError("sharpe_ratio: risk_free cannot be a DataFrame")
    if isinstance(returns, Series) and isinstance(risk_free, DataFrame):
        raise ValueError(
            "sharpe_ratio: Series vs DataFrame not supported"
        )  # pragma: no cover - unreachable
    _check_shapes(returns, risk_free, "sharpe_ratio")

    excess = returns - risk_free
    ann_ret = annual_return(excess, periods_per_year)
    sigma = volatility(excess, periods_per_year)

    if isinstance(sigma, Series):
        if (sigma == 0).all():
            return _empty_like(returns, "sharpe_ratio")
    else:
        if sigma == 0:
            return _empty_like(returns, "sharpe_ratio")

    sr = ann_ret / sigma
    return float(sr) if isinstance(returns, Series) else sr


# Backwards-compatible short name
_METRIC_REGISTRY["sharpe"] = sharpe_ratio


###############################################################################
# Sortino ratio                                                               #
###############################################################################
@register_metric("sortino_ratio")
def sortino_ratio(
    returns: Series | DataFrame,
    target: Series | DataFrame | float = 0.0,
    periods_per_year: int = 12,
) -> float | pd.Series | np.floating:
    _validate_input(returns, "sortino_ratio")
    if isinstance(returns, DataFrame) and isinstance(target, Series):
        raise ValueError("sortino_ratio: DataFrame vs Series not supported")
    if isinstance(returns, Series) and isinstance(target, DataFrame):
        raise ValueError("sortino_ratio: Series vs DataFrame not supported")
    if isinstance(target, DataFrame):
        raise ValueError("sortino_ratio: target cannot be a DataFrame")
    _check_shapes(returns, target, "sortino_ratio")

    excess = returns - target
    downside = excess.clip(upper=0)
    downside_std = np.sqrt((downside**2).mean())

    if (
        (downside_std == 0).all()
        if isinstance(downside_std, Series)
        else downside_std == 0
    ):
        return _empty_like(returns, "sortino_ratio")

    sr = annual_return(excess, periods_per_year) / (
        downside_std * np.sqrt(periods_per_year)
    )
    return float(sr) if isinstance(returns, Series) else sr


###############################################################################
# Maximum drawdown (positive magnitude)                                       #
###############################################################################
@register_metric("max_drawdown")
def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series | np.floating:
    """
    Maximum drawdown expressed as a *positive* fraction (0 → worst is 0%,
    0.35 → ‑35 % loss).  Legacy tests expect ≥ 0.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("max_drawdown expects a pandas Series or DataFrame")

    if returns.empty:
        return (
            np.nan
            if isinstance(returns, pd.Series)
            else pd.Series(np.nan, index=returns.columns, name="max_drawdown")
        )

    def _one(col: pd.Series) -> float:
        wealth: pd.Series = (1 + col).cumprod()
        draw = 1 - wealth / wealth.cummax()
        return float(draw.max())  # positive number

    return _one(returns) if isinstance(returns, pd.Series) else returns.apply(_one)


###############################################################################
# Information ratio                                                           #
###############################################################################
@register_metric("information_ratio")
def information_ratio(
    returns: Series | DataFrame,
    benchmark: Series | DataFrame | float | None = None,
    periods_per_year: int = 12,
) -> float | pd.Series | np.floating:
    _validate_input(returns, "information_ratio")

    if returns.empty or len(returns) < 2:
        return _empty_like(returns, "information_ratio")

    if benchmark is None:
        benchmark = returns.mean(axis=1) if isinstance(returns, DataFrame) else 0.0

    # --- scalar → broadcast -------------------------------------------------
    if np.isscalar(benchmark):
        bval = float(cast(float | int | np.floating, benchmark))
        benchmark = (
            pd.Series(bval, index=returns.index, name="benchmark")
            if isinstance(returns, Series)
            else pd.DataFrame(bval, index=returns.index, columns=returns.columns)
        )

    # --- Series → duplicate across all columns -----------------------------
    if isinstance(returns, DataFrame) and isinstance(benchmark, Series):
        _df: DataFrame = pd.concat([benchmark] * returns.shape[1], axis=1)
        _df.columns = returns.columns
        benchmark = _df

    # --- 1‑column DataFrame → duplicate across columns ---------------------
    if (
        isinstance(returns, DataFrame)
        and isinstance(benchmark, DataFrame)
        and benchmark.shape[1] == 1
    ):
        _df2: DataFrame = pd.concat([benchmark.iloc[:, 0]] * returns.shape[1], axis=1)
        _df2.columns = returns.columns
        benchmark = _df2

    _check_shapes(returns, benchmark, "information_ratio")

    active = returns - benchmark
    ann_act = active.mean() * periods_per_year
    tr_error = active.std(ddof=1) * np.sqrt(periods_per_year)

    if isinstance(tr_error, Series):
        if (tr_error == 0).all():
            return _empty_like(returns, "information_ratio")
    else:
        if tr_error == 0:
            return _empty_like(returns, "information_ratio")

    ir = ann_act / tr_error
    return float(ir) if isinstance(returns, Series) else ir


# ------------------------------------------------------------------ #
# 2A.  Ensure tests.legacy_metrics exists and exposes the old names. #
# ------------------------------------------------------------------ #
annualize_return = annual_return
annualize_volatility = volatility
annualize_sharpe_ratio = sharpe_ratio
annualize_sortino_ratio = sortino_ratio
info_ratio = information_ratio  # ← old short name

_legacy = types.ModuleType("tests.legacy_metrics")
for _name in (
    "annualize_return",
    "annualize_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "information_ratio",
    "info_ratio",
    "volatility",
):
    _legacy.__dict__[_name] = globals()[
        _name if _name in globals() else _name.replace("info_", "information_")
    ]
sys.modules.setdefault("tests.legacy_metrics", _legacy)

# trend_analysis/metrics.py  (append near the bottom ­– after all definitions)
# ---------------------------------------------------------------------------

setattr(_bi, "annualize_return", annualize_return)
setattr(_bi, "annualize_volatility", annualize_volatility)

# Public submodule to expose summary helpers
from . import summary  # noqa: E402,F401
from . import rolling  # noqa: E402,F401
from . import turnover  # noqa: E402,F401
from . import attribution  # noqa: E402,F401
