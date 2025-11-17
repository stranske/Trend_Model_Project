"""Signal generation primitives for trend-following strategies.

This module centralises the construction of strictly causal trading
signals so all execution paths (CLI, Streamlit UI, unit tests) rely on
the exact same implementation.  The helpers are intentionally vectorised
to keep the computation fast even for large universes.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import pandas as pd

SignalFrame: TypeAlias = pd.DataFrame


LOGGER = logging.getLogger(__name__)
_MEMO_ATTR = "_trend_signal_cache"


class _FrameHandle:
    __slots__ = ("frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get(self) -> pd.DataFrame:
        return self.frame

    def __deepcopy__(self, memo: dict[str, Any]) -> "_FrameHandle":
        return self


def _resolve_frame(entry: Any) -> pd.DataFrame | None:
    if isinstance(entry, pd.DataFrame):
        return entry
    if isinstance(entry, _FrameHandle):
        return entry.get()
    return None


def _ensure_signal_cache(frame: pd.DataFrame) -> dict[str, Any]:
    memo = frame.attrs.get(_MEMO_ATTR)
    if isinstance(memo, dict):
        return memo
    memo = {}
    frame.attrs[_MEMO_ATTR] = memo
    return memo


def _memoised_frame(
    frame: pd.DataFrame, key: str, builder: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    memo = _ensure_signal_cache(frame)
    cached = _resolve_frame(memo.get(key))
    if cached is not None:
        return cached
    numeric = builder()
    numeric.attrs[_MEMO_ATTR] = memo
    memo[key] = _FrameHandle(numeric)
    return numeric


def _memoised_rolling_stat(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    kind: Literal["mean", "std"],
) -> pd.DataFrame:
    memo = _ensure_signal_cache(frame)
    key = ("rolling", kind, window, min_periods)
    cached = _resolve_frame(memo.get(key))
    if cached is not None:
        return cached
    roller = frame.rolling(window=window, min_periods=min_periods)
    computed = roller.mean() if kind == "mean" else roller.std(ddof=0)
    memo[key] = _FrameHandle(computed)
    return computed


@contextmanager
def _timed_stage(stage: str):
    if not LOGGER.isEnabledFor(logging.DEBUG):
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.debug("compute_trend_signals[%s] %.2f ms", stage, duration_ms)


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
    def _build() -> pd.DataFrame:
        numeric = df.copy()
        for column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        return numeric.astype(float)

    return _memoised_frame(df, "float_frame", _build)


def _zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    demeaned = frame.sub(frame.mean(axis=1, skipna=True), axis=0)
    std = frame.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    normalised = demeaned.div(std, axis=0)
    return normalised.fillna(0.0)


def compute_trend_signals(returns: pd.DataFrame, spec: TrendSpec) -> pd.DataFrame:
    """Generate a strictly causal trend signal frame for the given returns."""

    if returns.empty:
        raise ValueError("returns cannot be empty")

    with _timed_stage("float_coerce"):
        numeric = _as_float_frame(returns)
    min_periods = spec.min_periods if spec.min_periods is not None else spec.window

    with _timed_stage("trend_mean"):
        rolling_mean = _memoised_rolling_stat(
            numeric, window=spec.window, min_periods=min_periods, kind="mean"
        )
    signal = rolling_mean.shift(spec.lag)

    if spec.vol_adjust:
        with _timed_stage("trend_vol"):
            rolling_std = _memoised_rolling_stat(
                numeric, window=spec.window, min_periods=min_periods, kind="std"
            ).shift(spec.lag)
        with np.errstate(divide="ignore", invalid="ignore"):
            if spec.vol_target is not None:
                scale = spec.vol_target / rolling_std
            else:
                scale = 1.0 / rolling_std
        scale = scale.replace([np.inf, -np.inf], np.nan)
        signal = signal.mul(scale)

    if spec.zscore:
        with _timed_stage("trend_zscore"):
            signal = _zscore_rows(signal)

    signal = signal.replace([np.inf, -np.inf], np.nan).astype(float)
    signal.attrs["spec"] = asdict(spec)
    signal.attrs["lag"] = spec.lag
    signal.attrs["kind"] = spec.kind
    return signal


__all__ = ["TrendSpec", "SignalFrame", "compute_trend_signals"]
