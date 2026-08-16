"""Signal generation primitives for trend-following strategies.

This module centralises the construction of strictly causal trading
signals so all execution paths (CLI, Streamlit UI, unit tests) rely on
the exact same implementation.  The helpers are intentionally vectorised
to keep the computation fast even for large universes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable, Literal, TypeAlias, cast

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

    def __deepcopy__(self, memo: dict[Any, Any]) -> "_FrameHandle":
        return self


def _resolve_frame(entry: Any) -> pd.DataFrame | None:
    if isinstance(entry, pd.DataFrame):
        return entry
    if isinstance(entry, _FrameHandle):
        return entry.get()
    return None


def _ensure_signal_cache(frame: pd.DataFrame) -> dict[Hashable, Any]:
    memo = frame.attrs.get(_MEMO_ATTR)
    if isinstance(memo, dict):
        return cast(dict[Hashable, Any], memo)
    new_memo: dict[Hashable, Any] = {}
    frame.attrs[_MEMO_ATTR] = new_memo
    return new_memo


def _memoised_frame(
    frame: pd.DataFrame,
    key: Hashable,
    builder: Callable[[], pd.DataFrame],
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
    key: tuple[str, Literal["mean", "std"], int, int] = (
        "rolling",
        kind,
        window,
        min_periods,
    )
    cached = _resolve_frame(memo.get(key))
    if cached is not None:
        return cached
    roller = frame.rolling(window=window, min_periods=min_periods)
    computed = roller.mean() if kind == "mean" else roller.std(ddof=0)
    memo[key] = _FrameHandle(computed)
    return computed


@contextmanager
def _timed_stage(stage: str) -> Iterator[None]:
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
    zscore: bool | float = False

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be a positive integer")
        if self.min_periods is not None and self.min_periods <= 0:
            raise ValueError("min_periods must be positive when provided")
        if self.lag < 1:
            raise ValueError("lag must be at least 1")
        if self.vol_target is not None and self.vol_target < 0:
            raise ValueError("vol_target must be non-negative when provided")


def _config_value(section: Any, key: str, default: Any = None) -> Any:
    """Read a signal setting from mappings, models, or lightweight config objects."""

    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    getter = getattr(section, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except KeyError:
                return default
        except KeyError:
            return default
    return getattr(section, key, default)


def trend_spec_from_mapping(
    signals: Any,
    *,
    vol_adjust: Any = None,
    retain_disabled_vol_target: bool = False,
) -> TrendSpec:
    """Build the shared signal contract from canonical keys."""

    if isinstance(signals, Mapping):
        removed_aliases = {
            "trend_" + "window": "window",
            "trend_" + "lag": "lag",
            "trend_" + "min_periods": "min_periods",
            "trend_" + "zscore": "zscore",
            "trend_" + "vol_adjust": "vol_adjust",
            "trend_" + "vol_target": "vol_target",
        }
        for key, replacement in removed_aliases.items():
            if key in signals:
                raise ValueError(f"signals.{key} was removed; use signals.{replacement}")

    def setting(key: str, default: Any = None) -> Any:
        value = _config_value(signals, key)
        return default if value is None else value

    kind = str(_config_value(signals, "kind", "tsmom") or "tsmom").lower()
    if kind != "tsmom":  # pragma: no cover - future extension guard
        raise ValueError(f"Unsupported trend signal kind: {kind}")

    try:
        window = int(setting("window", 63))
    except (TypeError, ValueError, OverflowError):
        window = 63
    try:
        min_periods_raw = setting("min_periods")
        min_periods = int(min_periods_raw) if min_periods_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        min_periods = None
    if min_periods is not None and min_periods <= 0:
        min_periods = None
    try:
        lag = max(1, int(setting("lag", 1)))
    except (TypeError, ValueError, OverflowError):
        lag = 1

    vol_adjust_flag = bool(setting("vol_adjust", _config_value(vol_adjust, "enabled", False)))
    vol_target_raw = setting("vol_target")
    if vol_target_raw is None and vol_adjust_flag:
        vol_target_raw = _config_value(vol_adjust, "target_vol")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
        if vol_target is not None and (not np.isfinite(vol_target) or vol_target <= 0):
            vol_target = None
    except (TypeError, ValueError):
        vol_target = None
    if not vol_adjust_flag and not retain_disabled_vol_target:
        vol_target = None

    zscore_setting = setting("zscore", False)
    if isinstance(zscore_setting, bool):
        zscore: bool | float = zscore_setting
    else:
        try:
            zscore_value = float(zscore_setting)
        except (TypeError, ValueError):
            zscore = False
        else:
            zscore = zscore_value if np.isfinite(zscore_value) and zscore_value > 0 else False

    return TrendSpec(
        kind="tsmom",
        window=max(1, window),
        min_periods=min_periods,
        lag=lag,
        vol_adjust=vol_adjust_flag,
        vol_target=vol_target,
        zscore=zscore,
    )


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


def _resolve_zscore_scale(setting: bool | float) -> float | None:
    if isinstance(setting, bool):
        return 1.0 if setting else None
    try:
        scale = float(setting)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale


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

    zscore_scale = _resolve_zscore_scale(spec.zscore)
    if zscore_scale is not None:
        with _timed_stage("trend_zscore"):
            signal = _zscore_rows(signal)
        if zscore_scale != 1.0:
            signal = signal.mul(zscore_scale)

    signal = signal.replace([np.inf, -np.inf], np.nan).astype(float)
    signal.attrs["spec"] = asdict(spec)
    signal.attrs["lag"] = spec.lag
    signal.attrs["kind"] = spec.kind
    return signal


__all__ = ["TrendSpec", "SignalFrame", "compute_trend_signals", "trend_spec_from_mapping"]
