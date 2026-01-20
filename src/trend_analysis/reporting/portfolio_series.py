"""Shared helpers for selecting portfolio return series from run results."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence, overload

import pandas as pd

_PORTFOLIO_SERIES_KEYS = (
    "portfolio_user_weight_combined",
    "portfolio_user_weight",
    "portfolio_equal_weight_combined",
    "portfolio_equal_weight",
)


def _series_is_empty(series: pd.Series) -> bool:
    return bool(series.empty) or bool(series.dropna().empty)


def _coerce_series(value: Any) -> pd.Series | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        series = value.copy()
        if _series_is_empty(series):
            return None
        try:
            return series.astype(float)
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        if "series" in value:
            nested = _coerce_series(value.get("series"))
            if nested is not None:
                return nested
        try:
            series = pd.Series(dict(value), dtype=float)
        except (TypeError, ValueError):
            return None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        series = pd.Series(list(value), dtype=float)
    else:
        series_like = getattr(value, "series", None)
        if isinstance(series_like, pd.Series) and not _series_is_empty(series_like):
            try:
                return series_like.copy().astype(float)
            except (TypeError, ValueError):
                return None
        return None
    if _series_is_empty(series):
        return None
    try:
        return series.astype(float)
    except (TypeError, ValueError):
        return None


def _normalise_weights(weights: Mapping[str, float]) -> pd.Series:
    series = pd.Series({str(k): float(v) for k, v in weights.items()})
    series = series.replace([math.inf, -math.inf], math.nan).dropna()
    series = series[series.abs() > 0]
    total = float(series.sum())
    if total:
        series = series / total
    return series


def _weighted_portfolio(
    out_df: pd.DataFrame | None,
    weights: Mapping[str, float] | None,
) -> pd.Series | None:
    if out_df is None or out_df.empty:
        return None
    if weights is not None:
        series = _normalise_weights(weights)
        if not series.empty:
            aligned = series.reindex(out_df.columns, fill_value=0.0)
            return out_df.mul(aligned, axis=1).sum(axis=1)
    if not len(out_df.columns):
        return None
    equal_weight = pd.Series(1.0 / float(len(out_df.columns)), index=out_df.columns)
    return out_df.mul(equal_weight, axis=1).sum(axis=1)


@overload
def select_primary_portfolio_series(
    res: Mapping[str, Any], *, prefer_raw: Literal[False] = False
) -> pd.Series | None: ...


@overload
def select_primary_portfolio_series(
    res: Mapping[str, Any], *, prefer_raw: Literal[True]
) -> Any | None: ...


def select_primary_portfolio_series(
    res: Mapping[str, Any], *, prefer_raw: bool = False
) -> pd.Series | Any | None:
    """Select the preferred portfolio series from a run result payload."""
    for key in _PORTFOLIO_SERIES_KEYS:
        raw_value = res.get(key)
        series = _coerce_series(raw_value)
        if series is not None:
            return raw_value if prefer_raw else series

    out_df = res.get("out_sample_scaled")
    out_df = out_df if isinstance(out_df, pd.DataFrame) else None
    weights = res.get("fund_weights")
    if not isinstance(weights, Mapping):
        weights = None
    if weights is None:
        weights = res.get("ew_weights")
        if not isinstance(weights, Mapping):
            weights = None

    return _weighted_portfolio(out_df, weights)
