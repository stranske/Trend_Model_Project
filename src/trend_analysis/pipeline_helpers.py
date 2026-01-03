from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .regimes import compute_regimes, normalise_settings
from .signals import TrendSpec
from .stages.preprocessing import _PreprocessStage, _WindowStage

logger = logging.getLogger("trend_analysis.pipeline")

# Default multiplier for reducing fund count/selection during risk-off regimes.
# Must match the value in config/defaults.yml regime.risk_off_fund_count_multiplier
_DEFAULT_RISK_OFF_FUND_MULTIPLIER = 0.5

__all__ = [
    "_apply_regime_overrides",
    "_apply_regime_weight_overrides",
    "_attach_calendar_settings",
    "_build_trend_spec",
    "_cfg_section",
    "_cfg_value",
    "_derive_split_from_periods",
    "_empty_run_full_result",
    "_format_period",
    "_policy_from_config",
    "_resolve_regime_label",
    "_resolve_sample_split",
    "_resolve_target_vol",
    "_section_get",
    "_unwrap_cfg",
    "compute_signal",
    "position_from_signal",
]


def _cfg_value(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_section(cfg: Mapping[str, Any] | Any, key: str) -> Any:
    section = _cfg_value(cfg, key, None)
    if section is None:
        return {}
    return section


def _section_get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, Mapping):
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
    attr_value = getattr(section, key, default)
    return attr_value


def _resolve_target_vol(vol_adjust_cfg: Mapping[str, Any] | Any) -> float | None:
    enabled = _section_get(vol_adjust_cfg, "enabled")
    if enabled is False:
        return None
    target_raw = _section_get(vol_adjust_cfg, "target_vol", 1.0)
    if target_raw is None:
        return 1.0
    try:
        target = float(target_raw)
    except (TypeError, ValueError):
        return 1.0
    if target <= 0:
        return None
    return target


def _resolve_regime_label(
    preprocess: _PreprocessStage,
    window: _WindowStage,
    regime_cfg: Mapping[str, Any] | None,
    benchmarks: Mapping[str, str] | None = None,
) -> tuple[str | None, Any]:
    def _resolve_proxy_column(proxy_value: str) -> str | None:
        columns = window.in_df.columns
        if proxy_value in columns:
            return proxy_value

        proxy_lower = proxy_value.lower()
        if benchmarks:
            for key, value in benchmarks.items():
                if proxy_lower == str(key).lower() and value in columns:
                    return str(value)
                if proxy_lower == str(value).lower() and value in columns:
                    return str(value)

        for col in columns:
            if proxy_lower == str(col).lower():
                return str(col)
        return None

    settings = normalise_settings(regime_cfg)
    if not settings.enabled:
        return None, settings
    proxy = settings.proxy
    if not proxy:
        return None, settings

    proxy_column = _resolve_proxy_column(proxy)
    if not proxy_column:
        return None, settings

    proxy_series = window.in_df[proxy_column].astype(float).dropna()
    if proxy_series.empty:
        return None, settings
    labels = compute_regimes(
        proxy_series,
        settings,
        freq=preprocess.freq_summary.target,
        periods_per_year=window.periods_per_year,
    )
    if labels.empty:
        return None, settings
    aligned = labels.reindex(window.in_df.index).ffill().bfill()
    if aligned.empty:
        return None, settings
    return str(aligned.iloc[-1]), settings


def _apply_regime_overrides(
    *,
    random_n: int,
    rank_kwargs: Mapping[str, Any] | None,
    regime_label: str | None,
    settings: Any,
    regime_cfg: Mapping[str, Any] | None = None,
) -> tuple[int, Mapping[str, Any] | None]:
    if not regime_label:
        return random_n, rank_kwargs
    if regime_label != getattr(settings, "risk_off_label", "Risk-Off"):
        return random_n, rank_kwargs

    # Get multiplier from config or use default
    cfg = dict(regime_cfg or {})
    multiplier = _DEFAULT_RISK_OFF_FUND_MULTIPLIER
    if "risk_off_fund_count_multiplier" in cfg:
        try:
            multiplier = float(cfg["risk_off_fund_count_multiplier"])
            if multiplier <= 0 or multiplier > 1:
                multiplier = _DEFAULT_RISK_OFF_FUND_MULTIPLIER
        except (TypeError, ValueError):
            pass

    updated_random_n = random_n
    if isinstance(random_n, int) and random_n > 1:
        updated_random_n = max(1, int(round(random_n * multiplier)))

    updated_rank = dict(rank_kwargs or {})
    inclusion_approach = str(updated_rank.get("inclusion_approach", "") or "").lower()
    if "n" in updated_rank and updated_rank.get("n") is not None:
        try:
            current_n = int(updated_rank["n"])
        except (TypeError, ValueError):
            current_n = None
        if current_n is not None and current_n > 1:
            updated_rank["n"] = max(1, int(round(current_n * multiplier)))

    if inclusion_approach == "top_pct" and updated_rank.get("pct") is not None:
        try:
            current_pct = float(updated_rank["pct"])
        except (TypeError, ValueError):
            current_pct = None
        if current_pct is not None and current_pct > 0:
            adjusted_pct = min(1.0, max(0.0, current_pct * multiplier))
            updated_rank["pct"] = adjusted_pct

    if inclusion_approach == "threshold" and updated_rank.get("threshold") is not None:
        try:
            current_threshold = float(updated_rank["threshold"])
        except (TypeError, ValueError):
            current_threshold = None
        if current_threshold is not None:
            if current_threshold > 0:
                updated_rank["threshold"] = current_threshold / multiplier
            elif current_threshold < 0:
                updated_rank["threshold"] = current_threshold * multiplier

    return updated_random_n, updated_rank if updated_rank else rank_kwargs


def _apply_regime_weight_overrides(
    *,
    target_vol: float | None,
    constraints: dict[str, Any] | None,
    regime_label: str | None,
    settings: Any,
    regime_cfg: Mapping[str, Any] | None,
) -> tuple[float | None, dict[str, Any] | None]:
    if target_vol is None:
        return None, constraints
    if not regime_label:
        return target_vol, constraints
    if regime_label != getattr(settings, "risk_off_label", "Risk-Off"):
        return target_vol, constraints

    cfg = dict(regime_cfg or {})

    def _coerce_positive_float(value: Any, default: float) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        if num <= 0:
            return default
        return num

    updated_target = target_vol
    if "risk_off_target_vol" in cfg:
        updated_target = _coerce_positive_float(
            cfg.get("risk_off_target_vol"), target_vol
        )
    else:
        multiplier = _coerce_positive_float(
            cfg.get("risk_off_target_vol_multiplier", 0.5), 0.5
        )
        updated_target = float(target_vol) * multiplier

    return updated_target, constraints


def _unwrap_cfg(cfg: Mapping[str, Any] | Any) -> Any:
    current = cfg
    visited: set[int] = set()
    while isinstance(current, Mapping) and "__cfg__" in current:
        marker = id(current)
        if marker in visited:  # pragma: no cover - defensive cycle guard
            break
        visited.add(marker)
        candidate = current.get("__cfg__")
        if candidate is None:
            break
        current = candidate
    return current


def _empty_run_full_result() -> dict[str, object]:
    return {
        "out_sample_stats": {},
        "in_sample_stats": {},
        "benchmark_ir": {},
        "risk_diagnostics": {},
        "fund_weights": {},
    }


def _build_trend_spec(
    cfg: Mapping[str, Any] | Any,
    vol_adjust_cfg: Mapping[str, Any] | Any,
) -> TrendSpec | None:
    """Build a TrendSpec from config, or None if no signals config is present.

    Returns None when no ``signals`` section is configured, so that callers
    fall back to the existing default behaviour (signal window derived from
    the in-sample risk window).
    """
    signals_cfg = _cfg_section(cfg, "signals")

    # Return None when no signals config is present - this preserves the old
    # default behaviour where signal_spec=None causes the pipeline to derive
    # the window from the risk window length rather than hardcoding 63.
    if not signals_cfg:
        return None

    def _signal_setting(key: str, alias: str | None, default: Any = None) -> Any:
        value = _section_get(signals_cfg, key, None)
        if value is None and alias:
            value = _section_get(signals_cfg, alias, None)
        return default if value is None else value

    kind = str(_section_get(signals_cfg, "kind", "tsmom") or "tsmom").lower()
    if kind != "tsmom":  # pragma: no cover - future extension guard
        raise ValueError(f"Unsupported trend signal kind: {kind}")

    try:
        window_raw = _signal_setting("window", "trend_window", 63)
        window = int(window_raw)
    except (TypeError, ValueError):
        window = 63
    min_periods_raw = _signal_setting("min_periods", "trend_min_periods")
    try:
        min_periods = int(min_periods_raw) if min_periods_raw is not None else None
    except (TypeError, ValueError):
        min_periods = None

    try:
        lag_raw = _signal_setting("lag", "trend_lag", 1)
        lag = max(1, int(lag_raw))
    except (TypeError, ValueError):
        lag = 1

    vol_adjust_default = bool(_section_get(vol_adjust_cfg, "enabled", False))
    vol_adjust_flag = bool(
        _signal_setting("vol_adjust", "trend_vol_adjust", vol_adjust_default)
    )
    vol_target_raw = _signal_setting("vol_target", "trend_vol_target")
    if vol_target_raw is None and vol_adjust_flag:
        vol_target_raw = _section_get(vol_adjust_cfg, "target_vol")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
        if vol_target is not None and vol_target <= 0:
            vol_target = None
    except (TypeError, ValueError):
        vol_target = None

    zscore_setting = _signal_setting("zscore", "trend_zscore", False)
    if isinstance(zscore_setting, bool):
        zscore_flag: bool | float = zscore_setting
    else:
        try:
            zscore_value = float(zscore_setting)
        except (TypeError, ValueError):
            zscore_flag = False
        else:
            zscore_flag = zscore_value if np.isfinite(zscore_value) else False
            if isinstance(zscore_flag, float) and zscore_flag <= 0:
                zscore_flag = False

    return TrendSpec(
        kind="tsmom",
        window=max(1, window),
        min_periods=min_periods,
        lag=lag,
        vol_adjust=vol_adjust_flag,
        vol_target=vol_target,
        zscore=zscore_flag,
    )


def _policy_from_config(
    cfg: Mapping[str, Any] | None,
) -> tuple[str | Mapping[str, str] | None, int | Mapping[str, int | None] | None]:
    if not cfg:
        return None, None
    policy_base = cfg.get("policy")
    per_asset = cfg.get("per_asset")
    policy_spec: str | Mapping[str, str] | None
    if isinstance(per_asset, Mapping):
        policy_spec = {str(k): str(v) for k, v in per_asset.items()}
        if policy_base is not None:
            policy_spec = {"default": str(policy_base), **policy_spec}
    elif policy_base is not None:
        policy_spec = str(policy_base)
    else:
        policy_spec = None

    limit_base = cfg.get("limit")
    per_asset_limit = cfg.get("per_asset_limit")
    if isinstance(per_asset_limit, Mapping):
        limit_map: dict[str, int | None] = {
            str(k): v for k, v in per_asset_limit.items()
        }
        if limit_base is not None:
            limit_map = {"default": limit_base, **limit_map}
        limit_spec: Mapping[str, int | None] | None = limit_map
    else:
        limit_spec = limit_base
    return policy_spec, limit_spec


def _format_period(period: pd.Period) -> str:
    return f"{period.year:04d}-{period.month:02d}"


def _derive_split_from_periods(
    periods: pd.PeriodIndex,
    *,
    method: str,
    boundary: pd.Period | None,
    ratio: float,
) -> dict[str, str]:
    if len(periods) == 0:
        raise ValueError("Unable to derive sample splits without any observations")
    if len(periods) == 1:
        period = periods[0]
        formatted = _format_period(period)
        return {
            "in_start": formatted,
            "in_end": formatted,
            "out_start": formatted,
            "out_end": formatted,
        }

    # Attempt date-based split first when boundary provided.
    if method == "date" and boundary is not None:
        in_mask = periods <= boundary
        out_mask = periods > boundary
        if in_mask.any() and out_mask.any():
            in_periods = periods[in_mask]
            out_periods = periods[out_mask]
            return {
                "in_start": _format_period(in_periods[0]),
                "in_end": _format_period(in_periods[-1]),
                "out_start": _format_period(out_periods[0]),
                "out_end": _format_period(out_periods[-1]),
            }

    # Fallback to ratio-based split when date split is unavailable or invalid.
    try:
        ratio_val = float(ratio)
    except (TypeError, ValueError):
        ratio_val = 0.7
    if not np.isfinite(ratio_val) or ratio_val <= 0:
        ratio_val = 0.5
    if ratio_val >= 1:
        ratio_val = 0.9
    in_count = int(round(len(periods) * ratio_val))
    if in_count <= 0:
        in_count = 1
    if in_count >= len(periods):
        in_count = len(periods) - 1

    in_periods = periods[:in_count]
    out_periods = periods[in_count:]
    if len(out_periods) == 0:
        raise ValueError("Unable to derive out-of-sample window from ratio split")
    return {
        "in_start": _format_period(in_periods[0]),
        "in_end": _format_period(in_periods[-1]),
        "out_start": _format_period(out_periods[0]),
        "out_end": _format_period(out_periods[-1]),
    }


def _resolve_sample_split(
    df: pd.DataFrame,
    split_cfg: Mapping[str, Any] | Any,
) -> dict[str, str]:
    required_keys = ("in_start", "in_end", "out_start", "out_end")
    resolved: dict[str, str] = {}
    for key in required_keys:
        value = _section_get(split_cfg, key)
        if value not in (None, ""):
            resolved[key] = str(value)

    missing = [key for key in required_keys if key not in resolved]
    if not missing:
        return resolved

    if "Date" not in df.columns:
        raise ValueError(
            "Input data must contain a 'Date' column to derive sample splits"
        )

    date_series = pd.to_datetime(df["Date"], errors="coerce")
    date_series = date_series.dropna()
    if date_series.empty:
        raise ValueError("Input data contains no valid dates to derive sample splits")

    sorted_periods = date_series.dt.to_period("M").sort_values()
    periods = pd.PeriodIndex(sorted_periods.unique())

    method_raw = _section_get(split_cfg, "method", "date")
    method = str(method_raw or "date").lower()
    boundary: pd.Period | None = None
    if method == "date":
        raw_boundary = _section_get(split_cfg, "date")
        if raw_boundary not in (None, ""):
            try:
                boundary = pd.Period(str(raw_boundary), freq="M")
            except Exception:
                boundary = None
    ratio_value = _section_get(split_cfg, "ratio", 0.7)

    derived = _derive_split_from_periods(
        periods,
        method=method,
        boundary=boundary,
        ratio=ratio_value,
    )

    for key, value in derived.items():
        resolved.setdefault(key, value)

    still_missing = [key for key in required_keys if key not in resolved]
    if still_missing:
        raise ValueError(
            f"Unable to derive sample split values for: {', '.join(still_missing)}"
        )
    return resolved


def compute_signal(
    df: pd.DataFrame,
    *,
    column: str = "returns",
    window: int = 3,
    min_periods: int | None = None,
    get_cache_func: Any | None = None,
    compute_dataset_hash_func: Any | None = None,
    log: logging.Logger | None = None,
) -> pd.Series:
    """Return a trailing rolling-mean signal using information strictly prior
    to the current row.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        column (str, optional): Name of the column to compute the signal from.
            Defaults to "returns".
        window (int, optional): Size of the trailing window for the rolling mean.
            Must be positive. Defaults to 3.
        min_periods (int or None, optional): Minimum number of observations in window
            required to have a value (otherwise result is NaN). If None, defaults to
            the value of `window`.
        get_cache_func: Optional cache accessor override (used for patching).
        compute_dataset_hash_func: Optional hash accessor override.
        log: Logger override used for debug messages.

    Returns:
        pd.Series: A float Series containing the strictly causal (look‑back only)
            rolling mean of the specified column. Value at index t uses rows
            (t-window) .. (t-1) and never the current row, enforcing shift‑safe
            behaviour. The Series is named "<column>_signal". NaN values appear
            until at least `min_periods` prior observations exist (plus one for
            the shift). Dtype is float.
    """

    if column not in df.columns:
        raise KeyError(column)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    base = df[column].astype(float)
    effective_min_periods = window if min_periods is None else int(min_periods)
    if effective_min_periods <= 0:
        raise ValueError("min_periods must be positive")

    def _compute() -> pd.Series:
        signal = (
            base.rolling(window=window, min_periods=effective_min_periods)
            .mean()
            .shift(1)
            .rename(f"{column}_signal")
        )
        return signal.astype(float)

    if get_cache_func is None:
        from .perf.rolling_cache import get_cache as get_cache_func  # type: ignore[assignment]
    if compute_dataset_hash_func is None:
        from .perf.rolling_cache import (
            compute_dataset_hash as compute_dataset_hash_func,  # type: ignore[assignment]
        )
    logger_to_use = log or logger

    try:
        cache = get_cache_func()
    except Exception:  # pragma: no cover - defensive
        return _compute()

    if not getattr(cache, "is_enabled", lambda: False)():
        return _compute()

    # For small series, the rolling mean is cheap and a filesystem-backed cache
    # can add overhead (hashing + IO) that dominates runtime and makes
    # property-based tests flaky on constrained runners.
    #
    # Still call into non-filesystem caches (e.g. test doubles) so cache
    # integration can be verified independently.
    try:
        from .perf.rolling_cache import RollingCache as _RollingCache

        if isinstance(cache, _RollingCache) and len(base) < 256:
            return _compute()
    except Exception:  # pragma: no cover - best-effort guard
        pass

    # Avoid relying on index .freq (tests may supply an index that raises).
    freq = "unknown"
    try:
        freq_str = getattr(df.index, "freqstr", None)
        if isinstance(freq_str, str) and freq_str:
            freq = freq_str
    except Exception as exc:  # pragma: no cover - best-effort guard
        logger_to_use.debug(
            "Failed to read index.freqstr when computing signal cache key: %s",
            exc,
        )

    dataset_hash = compute_dataset_hash_func([base])
    method = f"compute_signal:{column}:min{effective_min_periods}"
    return cache.get_or_compute(dataset_hash, window, freq, method, _compute)


def position_from_signal(
    signal: pd.Series,
    *,
    long_position: float = 1.0,
    short_position: float = -1.0,
    neutral_position: float = 0.0,
) -> pd.Series:
    """Convert a trading signal into positions using only past information.

    This function maps a time series of trading signals to position values, using only
    information available up to each point in time (no look-ahead bias).

    Rules:
        - The initial position is set to `neutral_position`.
        - For each signal value:
            - If the value is NaN or exactly zero, the position retains its previous value.
            - If the value is positive, the position is set to `long_position`.
            - If the value is negative, the position is set to `short_position`.
    Parameters:
        signal (pd.Series): The input trading signal.
        long_position (float, optional): Position value for positive signals (default: 1.0).
        short_position (float, optional): Position value for negative signals (default: -1.0).
        neutral_position (float, optional): Initial position and value for zero/NaN signals (default: 0.0).
    Returns:
        pd.Series: Series of position values, named "position", indexed as the input signal.
    """
    values = signal.astype(float).to_numpy()
    positions = np.empty_like(values, dtype=float)
    current = float(neutral_position)

    for idx, value in enumerate(values):
        if np.isnan(value) or value == 0.0:
            positions[idx] = current
            continue
        current = float(long_position if value > 0.0 else short_position)
        positions[idx] = current

    out = pd.Series(positions, index=signal.index, name="position")
    return out


def _attach_calendar_settings(df: pd.DataFrame, cfg: Any) -> None:
    preprocessing_section = _cfg_section(cfg, "preprocessing")
    data_settings = _cfg_section(cfg, "data")
    data_frequency = _section_get(data_settings, "frequency")
    data_timezone = _section_get(data_settings, "timezone", "UTC")
    holiday_calendar = _section_get(preprocessing_section, "holiday_calendar")
    df.attrs["calendar_settings"] = {
        "frequency": data_frequency,
        "timezone": data_timezone,
        "holiday_calendar": holiday_calendar,
    }
