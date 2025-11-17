"""Rolling aggregation helpers used across :mod:`trend_analysis`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar, cast

import pandas as pd


class _SupportsRolling(Protocol):
    """Protocol modelling pandas objects that expose ``rolling``."""

    def shift(
        self, periods: int
    ) -> "_SupportsRolling": ...  # pragma: no cover - Protocol

    def rolling(
        self,
        window: int,
        min_periods: int | None = None,
    ) -> pd.core.window.rolling.Rolling: ...  # pragma: no cover - Protocol


RollingAggregation = Callable[[pd.Series], float | int]
RollingLike = TypeVar("RollingLike", bound=_SupportsRolling)


def rolling_shifted(
    series: RollingLike,
    window: int,
    agg: str | RollingAggregation,
    min_periods: int | None = None,
) -> RollingLike:
    """Apply a rolling aggregation on prior-day values only.

    The helper enforces a one-period shift before computing rolling statistics,
    ensuring the resulting time-series never leaks same-day information.
    ``agg`` accepts built-in aggregations (mean, std, sum, max, min) or a
    callable which receives each rolling window as a :class:`pandas.Series`.
    """

    if window <= 0:
        raise ValueError("window must be a positive integer")
    if min_periods is not None and min_periods <= 0:
        raise ValueError("min_periods must be positive when provided")

    effective_min_periods = min_periods or window
    shifted = series.shift(1)
    rolling_obj = shifted.rolling(window=window, min_periods=effective_min_periods)

    if isinstance(agg, str):
        key = agg.strip().lower()
        if key == "mean":
            return cast(RollingLike, rolling_obj.mean())
        if key == "std":
            return cast(RollingLike, rolling_obj.std(ddof=0))
        if key == "sum":
            return cast(RollingLike, rolling_obj.sum())
        if key == "max":
            return cast(RollingLike, rolling_obj.max())
        if key == "min":
            return cast(RollingLike, rolling_obj.min())
        raise ValueError(
            "agg must be one of {'mean', 'std', 'sum', 'max', 'min'} when provided as a string"
        )

    if not callable(agg):  # pragma: no cover - defensive programming
        raise TypeError("agg must be a recognised string or callable")

    return cast(RollingLike, rolling_obj.apply(agg, raw=False))


__all__ = ["rolling_shifted"]
