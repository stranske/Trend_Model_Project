"""Shared risk-control helpers for volatility targeting and constraints.

This module centralises the lightweight risk controls requested in Issue
#1680 so both the command-line interface and the Streamlit application apply
identical logic.  The helpers deliberately operate on pandas objects to keep
callers ergonomic – the surrounding pipeline already works with DataFrames and
Series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np
import pandas as pd

NUMERICAL_EPS = 1e-9


@dataclass(slots=True)
class RiskConfig:
    """Configuration bundle for risk helpers.

    Parameters
    ----------
    target_vol:
        Target annualised volatility for each asset after scaling.
    floor_vol:
        Minimum realised volatility used as denominator when computing scale
        factors.  Prevents division by zero and excessive leverage when
        realised vol is tiny.
    warmup_periods:
        Number of leading rows to zero out after scaling.  Used to honour the
        "warm-up" requirement where exposures are ramped gradually from zero.
    lookback:
        Window length (in periods) used when computing realised volatility.
    annualisation:
        Periods per year for annualising the realised volatility calculation.
    long_only:
        When ``True`` negative weights are clipped to zero.
    max_weight:
        Optional per-position cap (post normalisation).
    sum_to_one:
        When ``True`` the helper renormalises weights so they add to one after
        constraints are applied.
    turnover_cap:
        Optional L1 turnover cap applied when comparing to prior weights.
    transaction_cost:
        Linear transaction cost deducted after volatility scaling (per asset,
        same implementation as the previous pipeline logic).
    """

    target_vol: float = 0.10
    floor_vol: float | None = None
    warmup_periods: int = 0
    lookback: int = 36
    annualisation: float = 12.0
    long_only: bool = False
    max_weight: float | None = None
    sum_to_one: bool = True
    turnover_cap: float | None = None
    transaction_cost: float = 0.0


@dataclass(slots=True)
class RiskDiagnostics:
    """Container for diagnostic series."""

    realized_vol: pd.DataFrame
    turnover: pd.DataFrame
    scale_factors: pd.Series


@dataclass(slots=True)
class RiskResult:
    """Outputs of :func:`apply_risk_controls`."""

    in_scaled: pd.DataFrame
    out_scaled: pd.DataFrame
    equal_weights: pd.Series
    user_weights: pd.Series
    diagnostics: RiskDiagnostics


def compute_realized_vol(
    returns: pd.DataFrame,
    *,
    window: int,
    annualisation: float,
) -> pd.DataFrame:
    """Compute realised volatility per asset.

    A trailing window standard deviation is annualised using
    ``sqrt(annualisation)``.  The function is robust to small ``window`` sizes –
    a minimum of one observation is requested so the first few rows are still
    populated.
    """

    if returns.empty:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    window = max(int(window), 1)
    rolling = returns.rolling(window=window, min_periods=1).std()
    realised = rolling * np.sqrt(float(annualisation))
    return realised.astype(float)


def _normalise(weights: pd.Series, *, sum_to_one: bool) -> pd.Series:
    if weights.empty:
        return weights
    if not sum_to_one:
        return weights
    total = float(weights.sum())
    if abs(total) <= NUMERICAL_EPS:
        return weights.fillna(0.0)
    return weights / total


def _apply_position_limits(weights: pd.Series, cfg: RiskConfig) -> pd.Series:
    work = weights.astype(float).copy()
    if cfg.long_only:
        work = work.clip(lower=0.0)
    if cfg.max_weight is not None:
        work = work.clip(upper=float(cfg.max_weight))
    work = _normalise(work, sum_to_one=cfg.sum_to_one)
    return work.fillna(0.0)


def _enforce_turnover(
    weights: pd.Series,
    prev_weights: pd.Series | None,
    cap: float | None,
    *,
    sum_to_one: bool,
) -> pd.Series:
    if prev_weights is None or cap is None or cap <= NUMERICAL_EPS:
        return weights

    aligned_prev = prev_weights.reindex(weights.index, fill_value=0.0)
    delta = weights - aligned_prev
    turnover = float(delta.abs().sum())
    if turnover <= cap + NUMERICAL_EPS:
        return weights

    scale = cap / turnover if turnover > 0 else 0.0
    adjusted = aligned_prev + delta * scale
    return _normalise(adjusted, sum_to_one=sum_to_one).clip(lower=0.0)


def _compute_turnover_series(
    *,
    prev_weights: pd.Series | None,
    new_weights: pd.Series,
    label: str,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    if prev_weights is None:
        return pd.DataFrame(columns=[label], dtype=float)

    aligned_prev = prev_weights.reindex(new_weights.index, fill_value=0.0)
    turnover = float((new_weights - aligned_prev).abs().sum())
    idx = pd.Index([timestamp]) if timestamp is not None else pd.RangeIndex(1)
    df = pd.DataFrame({label: [turnover]}, index=idx)
    df.index.name = "Date"
    return df


def apply_risk_controls(
    *,
    in_returns: pd.DataFrame,
    out_returns: pd.DataFrame,
    base_equal_weights: pd.Series,
    base_user_weights: pd.Series,
    cfg: RiskConfig,
    prev_equal_weights: pd.Series | None = None,
    prev_user_weights: pd.Series | None = None,
    turnover_timestamp: pd.Timestamp | None = None,
) -> RiskResult:
    """Apply volatility targeting and constraints to the provided returns."""

    cols = list(dict.fromkeys(base_user_weights.index))
    realised = compute_realized_vol(
        pd.concat([in_returns[cols], out_returns[cols]], axis=0),
        window=cfg.lookback,
        annualisation=cfg.annualisation,
    )

    if not realised.empty:
        last_in_idx = realised.index.intersection(in_returns.index)
        if not last_in_idx.empty:
            last_realised = realised.loc[last_in_idx[-1]]
        else:
            last_realised = realised.iloc[-1]
    else:
        last_realised = pd.Series(0.0, index=cols, dtype=float)

    floor = cfg.floor_vol if cfg.floor_vol is not None else 0.0
    denom = last_realised.clip(lower=max(float(floor), 0.0))
    denom.replace(0.0, np.nan, inplace=True)
    scale_factors = pd.Series(cfg.target_vol, index=cols, dtype=float).div(denom)
    scale_factors.replace([np.inf, -np.inf], 0.0, inplace=True)
    scale_factors = scale_factors.fillna(0.0)

    in_scaled = in_returns[cols].mul(scale_factors, axis=1)
    out_scaled = out_returns[cols].mul(scale_factors, axis=1)

    if cfg.transaction_cost:
        in_scaled = in_scaled.sub(float(cfg.transaction_cost))
        out_scaled = out_scaled.sub(float(cfg.transaction_cost))

    if cfg.warmup_periods > 0:
        warm = int(cfg.warmup_periods)
        if warm > 0:
            if not in_scaled.empty:
                in_scaled.iloc[: min(warm, len(in_scaled))] = 0.0
            if not out_scaled.empty:
                out_scaled.iloc[: min(warm, len(out_scaled))] = 0.0

    in_scaled = in_scaled.clip(lower=-1.0).fillna(0.0)
    out_scaled = out_scaled.clip(lower=-1.0).fillna(0.0)

    eq_weights = _apply_position_limits(base_equal_weights, cfg)
    user_weights = _apply_position_limits(base_user_weights, cfg)

    eq_weights = _apply_position_limits(
        _enforce_turnover(eq_weights, prev_equal_weights, cfg.turnover_cap, sum_to_one=cfg.sum_to_one),
        cfg,
    )
    user_weights = _apply_position_limits(
        _enforce_turnover(user_weights, prev_user_weights, cfg.turnover_cap, sum_to_one=cfg.sum_to_one),
        cfg,
    )

    turnover_frames = []
    eq_turn = _compute_turnover_series(
        prev_weights=prev_equal_weights,
        new_weights=eq_weights,
        label="equal_weight",
        timestamp=turnover_timestamp,
    )
    if not eq_turn.empty:
        turnover_frames.append(eq_turn)
    user_turn = _compute_turnover_series(
        prev_weights=prev_user_weights,
        new_weights=user_weights,
        label="user_weight",
        timestamp=turnover_timestamp,
    )
    if not user_turn.empty:
        if turnover_frames:
            turnover_frames[0] = turnover_frames[0].join(user_turn, how="outer")
        else:
            turnover_frames.append(user_turn)

    turnover_df = turnover_frames[0] if turnover_frames else pd.DataFrame(columns=["equal_weight", "user_weight"])

    diagnostics = RiskDiagnostics(
        realized_vol=realised,
        turnover=turnover_df,
        scale_factors=scale_factors,
    )

    return RiskResult(
        in_scaled=in_scaled,
        out_scaled=out_scaled,
        equal_weights=eq_weights,
        user_weights=user_weights,
        diagnostics=diagnostics,
    )


def summarize_turnover(turnover: pd.DataFrame) -> Mapping[str, float]:
    """Return simple aggregates (mean turnover per label)."""

    if turnover.empty:
        return {}
    summary: MutableMapping[str, float] = {}
    for column in turnover.columns:
        try:
            summary[column] = float(pd.to_numeric(turnover[column], errors="coerce").mean())
        except Exception:  # pragma: no cover - defensive
            continue
    return summary


__all__ = [
    "NUMERICAL_EPS",
    "RiskConfig",
    "RiskDiagnostics",
    "RiskResult",
    "apply_risk_controls",
    "compute_realized_vol",
    "summarize_turnover",
]

