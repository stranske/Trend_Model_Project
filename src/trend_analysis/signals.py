"""Signal generation primitives for trend-following strategies.

This module centralises the construction of strictly causal trading
signals so all execution paths (CLI, Streamlit UI, unit tests) rely on
the exact same implementation.  The helpers are intentionally vectorised
to keep the computation fast even for large universes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

SignalFrame: TypeAlias = pd.DataFrame


@dataclass(frozen=True, slots=True)
class TrendSpec:
    """Configuration for generating time-series momentum style signals."""

    kind: Literal["tsmom"] = "tsmom"
    window: int = 63
    min_periods: int | None = None
    lag: int = 1
    vol_adjust: bool = False
    vol_target: float | None = None
    zscore: bool = False

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be a positive integer")
        if self.min_periods is not None and self.min_periods <= 0:
            raise ValueError("min_periods must be positive when provided")
        if self.lag < 1:
            raise ValueError("lag must be at least 1")
        if self.vol_target is not None and self.vol_target < 0:
            raise ValueError("vol_target must be non-negative when provided")


def _as_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.astype(float)


def _zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    demeaned = frame.sub(frame.mean(axis=1, skipna=True), axis=0)
    std = frame.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    normalised = demeaned.div(std, axis=0)
    return normalised.fillna(0.0)


def compute_trend_signals(returns: pd.DataFrame, spec: TrendSpec) -> pd.DataFrame:
    """Generate a strictly causal trend signal frame for the given returns."""

    if returns.empty:
        raise ValueError("returns cannot be empty")

    numeric = _as_float_frame(returns)
    min_periods = spec.min_periods if spec.min_periods is not None else spec.window

    rolling_mean = numeric.rolling(window=spec.window, min_periods=min_periods).mean()
    signal = rolling_mean.shift(spec.lag)

    if spec.vol_adjust:
        rolling_std = (
            numeric.rolling(window=spec.window, min_periods=min_periods)
            .std(ddof=0)
            .shift(spec.lag)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            if spec.vol_target is not None:
                scale = spec.vol_target / rolling_std
            else:
                scale = 1.0 / rolling_std
        scale = scale.replace([np.inf, -np.inf], np.nan)
        signal = signal.mul(scale)

    if spec.zscore:
        signal = _zscore_rows(signal)

    signal = signal.replace([np.inf, -np.inf], np.nan).astype(float)
    signal.attrs["spec"] = asdict(spec)
    signal.attrs["lag"] = spec.lag
    signal.attrs["kind"] = spec.kind
    return signal


__all__ = ["TrendSpec", "SignalFrame", "compute_trend_signals"]
