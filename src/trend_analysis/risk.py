"""Shared risk-control helpers for volatility targeting and constraints.

This module centralises volatility targeting, weight normalisation, and
turnover enforcement so that both the CLI pipeline and the Streamlit app
operate with the exact same risk logic.  Functions are intentionally
stateless – callers supply the relevant slices of returns and any
previous-period weights and receive normalised, constraint-aware weights
alongside diagnostics that power reporting layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd

from .engine import optimizer as optimizer_mod

PERIODS_PER_YEAR: Mapping[str, float] = {
    "D": 252.0,
    "W": 52.0,
    "M": 12.0,
    "Q": 4.0,
    "Y": 1.0,
}


@dataclass(slots=True)
class RiskWindow:
    """Specification for realised-volatility lookback windows."""

    length: int
    decay: str = "simple"
    ewma_lambda: float = 0.94


@dataclass(slots=True)
class RiskDiagnostics:
    """Diagnostics emitted by :func:`compute_constrained_weights`."""

    asset_volatility: pd.DataFrame
    portfolio_volatility: pd.Series
    turnover: pd.Series
    turnover_value: float
    scale_factors: pd.Series


def periods_per_year_from_code(code: str | None) -> float:
    """Return periods-per-year scaling given a frequency code."""

    if not code:
        return 12.0
    return PERIODS_PER_YEAR.get(code.upper(), 12.0)


def _ensure_series(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.astype(float).copy()
    else:
        series = pd.Series({str(k): float(v) for k, v in weights.items()}, dtype=float)
    return series.sort_index()


def realised_volatility(
    returns: pd.DataFrame,
    window: RiskWindow,
    *,
    periods_per_year: float = 12.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute realised volatility per asset across ``returns``.

    The helper supports both simple rolling windows and an EWMA variant
    that mirrors common risk engines.
    """

    if returns.empty:
        raise ValueError("returns cannot be empty")

    rolling_kwargs: MutableMapping[str, object] = {
        "min_periods": min_periods or 1,
    }

    if window.length <= 0:
        raise ValueError("window length must be positive")

    returns = returns.astype(float)

    if window.decay.lower() == "ewma":
        lam = float(window.ewma_lambda)
        alpha = 1.0 - lam
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"ewma_lambda must be between 0 and 1 (got {lam}); "
                f"computed alpha = 1 - ewma_lambda = {alpha:.4f} must be between 0 and 1"
            )
        vol = returns.ewm(alpha=alpha, adjust=False).std(bias=False)
    else:
        vol = returns.rolling(window=window.length, **rolling_kwargs).std(ddof=0)

    return vol.mul(np.sqrt(periods_per_year))


def _scale_factors(
    latest_vol: pd.Series,
    target_vol: float,
    *,
    floor_vol: float | None = None,
) -> pd.Series:
    vol = latest_vol.astype(float).replace(0.0, np.nan)
    if floor_vol is not None and floor_vol > 0:
        vol = vol.clip(lower=float(floor_vol))
    factors = pd.Series(target_vol, index=vol.index, dtype=float).div(vol)
    factors = factors.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return factors


def _normalise(weights: pd.Series) -> pd.Series:
    total = float(weights.sum())
    if total == 0.0:
        return weights.copy()
    return weights / total


def _enforce_turnover_cap(
    target: pd.Series,
    prev: pd.Series | None,
    max_turnover: float | None,
) -> tuple[pd.Series, float]:
    target = target.astype(float)
    if prev is None:
        turnover = float(target.abs().sum())
        return target, turnover

    aligned_index = target.index.union(prev.index)
    prev_aligned = prev.reindex(aligned_index, fill_value=0.0)
    target_aligned = target.reindex(aligned_index, fill_value=0.0)
    trades = target_aligned - prev_aligned
    turnover = float(trades.abs().sum())
    if max_turnover is None or max_turnover <= 0 or turnover <= max_turnover:
        return target_aligned, turnover
    scale = max_turnover / turnover if turnover > 0 else 0.0
    adjusted = prev_aligned + trades * scale
    return adjusted, float((adjusted - prev_aligned).abs().sum())


def compute_constrained_weights(
    base_weights: Mapping[str, float] | pd.Series,
    returns: pd.DataFrame,
    *,
    window: RiskWindow,
    target_vol: float,
    periods_per_year: float,
    floor_vol: float | None,
    long_only: bool,
    max_weight: float | None,
    previous_weights: Mapping[str, float] | pd.Series | None = None,
    max_turnover: float | None = None,
    group_caps: Mapping[str, float] | None = None,
    groups: Mapping[str, str] | None = None,
) -> tuple[pd.Series, RiskDiagnostics]:
    """Apply risk controls and return final weights plus diagnostics."""

    if returns.empty:
        raise ValueError("returns cannot be empty")

    base = _ensure_series(base_weights)
    if base.empty:
        raise ValueError("base_weights cannot be empty")

    base = _normalise(base)
    realised = realised_volatility(returns, window, periods_per_year=periods_per_year)
    latest = realised.iloc[-1].reindex(base.index).ffill().bfill()
    latest = latest.reindex(base.index).fillna(realised.mean(axis=0))
    positive = latest[latest > 0]
    fallback = float(positive.min()) if not positive.empty else 1.0
    latest = latest.fillna(fallback)

    scale_factors = _scale_factors(latest, target_vol, floor_vol=floor_vol)
    scaled = base.mul(scale_factors)

    constraint_payload = {
        "long_only": bool(long_only),
        "max_weight": max_weight,
        "group_caps": group_caps,
        "groups": groups,
    }
    constrained = optimizer_mod.apply_constraints(scaled, constraint_payload)

    prev_series = (
        _ensure_series(previous_weights) if previous_weights is not None else None
    )
    constrained, turnover_value = _enforce_turnover_cap(
        constrained, prev_series, max_turnover
    )
    constrained = constrained.reindex(base.index, fill_value=0.0)
    constrained = _normalise(constrained)

    aligned_returns = returns.reindex(columns=constrained.index, fill_value=0.0)
    portfolio_returns = aligned_returns.mul(constrained, axis=1).sum(axis=1)
    portfolio_vol = realised_volatility(
        portfolio_returns.to_frame("portfolio"),
        window,
        periods_per_year=periods_per_year,
    )["portfolio"]

    turnover_index: Iterable[pd.Timestamp]
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) > 0:
        turnover_index = [returns.index[-1]]
    else:
        turnover_index = [pd.Timestamp("1970-01-01")]  # dummy timestamp

    turnover_series = pd.Series(
        [turnover_value],
        index=pd.Index(turnover_index, name="rebalance"),
        name="turnover",
    )

    diagnostics = RiskDiagnostics(
        asset_volatility=realised,
        portfolio_volatility=portfolio_vol,
        turnover=turnover_series,
        turnover_value=float(turnover_value),
        scale_factors=scale_factors.reindex(constrained.index).fillna(0.0),
    )

    return constrained, diagnostics


__all__ = [
    "RiskDiagnostics",
    "RiskWindow",
    "compute_constrained_weights",
    "periods_per_year_from_code",
    "realised_volatility",
]
