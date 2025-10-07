from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core.rank_selection import (
    RiskStatsConfig,
    get_window_metric_bundle,
    make_window_key,
    rank_select_funds,
)
from .data import load_csv
from .metrics import (
    annual_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)
from .perf.rolling_cache import compute_dataset_hash, get_cache
from .regimes import build_regime_payload
from .risk import (
    RiskDiagnostics,
    RiskWindow,
    compute_constrained_weights,
    periods_per_year_from_code,
    realised_volatility,
)
from .signals import TrendSpec, compute_trend_signals
from .timefreq import MONTHLY_DATE_FREQ
from .util.frequency import FrequencySummary, detect_frequency
from .util.missing import MissingPolicyResult, apply_missing_policy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as Config

del TYPE_CHECKING


@dataclass
class _Stats:
    """Container for performance metrics.

    AvgCorr fields are optional and only populated when the user explicitly
    requests the ``AvgCorr`` metric (Issue #1160). They remain ``None`` to
    preserve backward compatibility and avoid altering column order when the
    feature is not in use.
    """

    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    information_ratio: float
    is_avg_corr: float | None = None
    os_avg_corr: float | None = None


def _frequency_label(code: str) -> str:
    return {"D": "Daily", "W": "Weekly", "M": "Monthly"}.get(code, code)


def _preprocessing_summary(
    freq_code: str, *, normalised: bool, missing_summary: str | None
) -> str:
    cadence = _frequency_label(freq_code)
    cadence_text = f"Cadence: {cadence}"
    if normalised and freq_code != "M":
        cadence_text += " → monthly"
    elif freq_code == "M":
        cadence_text += " (month-end)"
    parts = [cadence_text]
    if missing_summary:
        parts.append(f"Missing data: {missing_summary}")
    return "; ".join(parts)


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
            except Exception:  # pragma: no cover - defensive
                return default
        except KeyError:
            return default
    return getattr(section, key, default)


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
) -> TrendSpec:
    signals_cfg = _cfg_section(cfg, "signals")
    kind = str(_section_get(signals_cfg, "kind", "tsmom") or "tsmom").lower()
    if kind != "tsmom":  # pragma: no cover - future extension guard
        raise ValueError(f"Unsupported trend signal kind: {kind}")

    try:
        window_raw = _section_get(signals_cfg, "window", 63)
        window = int(window_raw)
    except (TypeError, ValueError):
        window = 63
    min_periods_raw = _section_get(signals_cfg, "min_periods")
    try:
        min_periods = int(min_periods_raw) if min_periods_raw is not None else None
    except (TypeError, ValueError):
        min_periods = None

    try:
        lag_raw = _section_get(signals_cfg, "lag", 1)
        lag = max(1, int(lag_raw))
    except (TypeError, ValueError):
        lag = 1

    vol_adjust_default = bool(_section_get(vol_adjust_cfg, "enabled", False))
    vol_adjust_flag = bool(_section_get(signals_cfg, "vol_adjust", vol_adjust_default))
    vol_target_raw = _section_get(signals_cfg, "vol_target")
    if vol_target_raw is None and vol_adjust_flag:
        vol_target_raw = _section_get(vol_adjust_cfg, "target_vol")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
        if vol_target is not None and vol_target <= 0:
            vol_target = None
    except (TypeError, ValueError):
        vol_target = None

    zscore_flag = bool(_section_get(signals_cfg, "zscore", False))

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


def _prepare_input_data(
    df: pd.DataFrame,
    *,
    date_col: str,
    missing_policy: str | Mapping[str, str] | None,
    missing_limit: int | Mapping[str, int | None] | None,
    enforce_completeness: bool = True,
) -> tuple[pd.DataFrame, FrequencySummary, MissingPolicyResult, bool]:
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{date_col}' column")

    work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work[date_col].dtype):
        work[date_col] = pd.to_datetime(work[date_col])
    work.sort_values(date_col, inplace=True)

    freq_summary = detect_frequency(work[date_col])

    value_cols = [c for c in work.columns if c != date_col]
    if value_cols:
        numeric = work[value_cols].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = work[value_cols]
    numeric.index = pd.DatetimeIndex(work[date_col])

    if freq_summary.resampled:
        resampled = (1 + numeric).resample(MONTHLY_DATE_FREQ).prod(min_count=1) - 1
        normalised = True
    else:
        resampled = numeric.resample(MONTHLY_DATE_FREQ).last()
        normalised = False

    resampled = resampled.dropna(how="all")
    resampled.index.name = date_col

    policy_spec: str | Mapping[str, str] | None = missing_policy or "drop"
    filled, missing_result = apply_missing_policy(
        resampled,
        policy=policy_spec,
        limit=missing_limit,
        enforce_completeness=enforce_completeness,
    )
    filled = filled.dropna(how="all")

    if filled.empty:
        monthly_df = pd.DataFrame(columns=[date_col])
    else:
        monthly_df = filled.reset_index().rename(columns={"index": date_col})
        monthly_df[date_col] = pd.to_datetime(monthly_df[date_col])
        monthly_df.sort_values(date_col, inplace=True)

    return monthly_df, freq_summary, missing_result, normalised


def calc_portfolio_returns(
    weights: NDArray[Any], returns_df: pd.DataFrame
) -> pd.Series:
    """Calculate weighted portfolio returns."""
    return returns_df.mul(weights, axis=1).sum(axis=1)


def single_period_run(
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    stats_cfg: RiskStatsConfig | None = None,
) -> pd.DataFrame:
    """Return a score frame of metrics for a single period.

    Parameters
    ----------
    df : pd.DataFrame
        Input returns data with a ``Date`` column.
    start, end : str
        Inclusive period in ``YYYY-MM`` format.
    stats_cfg : RiskStatsConfig | None
        Metric configuration; defaults to ``RiskStatsConfig()``.

    Returns
    -------
    pd.DataFrame
        Table of metric values (index = fund code).  The frame is pure
        and carries ``insample_len`` and ``period`` metadata so callers
        can reason about the analysed window.
    """
    from .core.rank_selection import RiskStatsConfig, _compute_metric_series

    if stats_cfg is None:
        stats_cfg = RiskStatsConfig()

    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"].dtype):
        df["Date"] = pd.to_datetime(df["Date"])

    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    sdate, edate = _parse_month(start), _parse_month(end)
    window = df[(df["Date"] >= sdate) & (df["Date"] <= edate)].set_index("Date")

    metrics = stats_cfg.metrics_to_run
    if not metrics:
        raise ValueError("stats_cfg.metrics_to_run must not be empty")

    parts = [
        _compute_metric_series(window.dropna(axis=1, how="all"), m, stats_cfg)
        for m in metrics
    ]
    score_frame = pd.concat(parts, axis=1)
    score_frame.columns = metrics
    score_frame.attrs["insample_len"] = len(window)
    score_frame.attrs["period"] = (start, end)
    # Optional derived correlation metric (opt-in via stats_cfg.extra_metrics)
    try:
        extra = getattr(stats_cfg, "extra_metrics", [])
        if (
            "AvgCorr" in extra
            and score_frame.shape[1] > 0
            and window.shape[1] > 1
            and "AvgCorr" not in score_frame.columns
        ):
            from .core.rank_selection import compute_metric_series_with_cache

            avg_corr_series = compute_metric_series_with_cache(
                window.dropna(axis=1, how="all"),
                "AvgCorr",
                stats_cfg,
                enable_cache=False,
            )
            score_frame = pd.concat([score_frame, avg_corr_series], axis=1)
    except Exception:  # pragma: no cover - defensive
        pass
    return score_frame.astype(float)


def _compute_stats(
    df: pd.DataFrame,
    rf: pd.Series,
    *,
    in_sample_avg_corr: dict[str, float] | None = None,
    out_sample_avg_corr: dict[str, float] | None = None,
) -> dict[str, _Stats]:
    # Metrics expect 1D Series; iterating keeps the logic simple for a handful
    # of columns and avoids reshaping into higher-dimensional arrays.
    stats: dict[str, _Stats] = {}
    for col in df:
        key = str(col)
        stats[key] = _Stats(
            cagr=float(annual_return(df[col])),
            vol=float(volatility(df[col])),
            sharpe=float(sharpe_ratio(df[col], rf)),
            sortino=float(sortino_ratio(df[col], rf)),
            max_drawdown=float(max_drawdown(df[col])),
            information_ratio=float(information_ratio(df[col], rf)),
            is_avg_corr=(in_sample_avg_corr or {}).get(col),
            os_avg_corr=(out_sample_avg_corr or {}).get(col),
        )
    return stats


def _run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float,
    monthly_cost: float,
    *,
    floor_vol: float | None = None,
    warmup_periods: int = 0,
    selection_mode: str = "all",
    random_n: int = 8,
    custom_weights: dict[str, float] | None = None,
    rank_kwargs: Mapping[str, Any] | None = None,
    manual_funds: list[str] | None = None,
    indices_list: list[str] | None = None,
    benchmarks: dict[str, str] | None = None,
    seed: int = 42,
    stats_cfg: RiskStatsConfig | None = None,
    weighting_scheme: str | None = None,
    constraints: dict[str, Any] | None = None,
    missing_policy: str | Mapping[str, str] | None = None,
    missing_limit: int | Mapping[str, int | None] | None = None,
    risk_window: Mapping[str, Any] | None = None,
    periods_per_year_override: float | None = None,
    previous_weights: Mapping[str, float] | None = None,
    max_turnover: float | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
) -> dict[str, object] | None:
    if df is None:
        return None

    # Guard against negative configuration inputs.  ``floor_vol`` enforces the
    # minimum realised volatility used for scaling so we never divide by zero,
    # while ``warmup_periods`` zeroes the initial rows (Issue #1439).
    try:
        min_floor = float(floor_vol) if floor_vol is not None else 0.0
    except (TypeError, ValueError):  # pragma: no cover - defensive
        min_floor = 0.0
    if min_floor < 0:
        min_floor = 0.0
    try:
        warmup = int(warmup_periods)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        warmup = 0
    if warmup < 0:
        warmup = 0

    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    na_cfg = getattr(stats_cfg, "na_as_zero_cfg", None) if stats_cfg else None
    enforce_complete = not (na_cfg and bool(na_cfg.get("enabled", False)))

    df_prepared, freq_summary, missing_result, normalised = _prepare_input_data(
        df,
        date_col=date_col,
        missing_policy=missing_policy,
        missing_limit=missing_limit,
        enforce_completeness=enforce_complete,
    )
    if df_prepared.empty:
        return None

    df = df_prepared.copy()
    value_cols_all = [c for c in df.columns if c != date_col]
    if not value_cols_all:
        return None

    if df.empty or df.shape[1] <= 1:
        return None

    freq_code = freq_summary.code
    missing_meta = missing_result

    frequency_payload = {
        "code": freq_summary.code,
        "label": freq_summary.label,
        "target": freq_summary.target,
        "target_label": freq_summary.target_label,
        "resampled": freq_summary.resampled,
    }
    periods_per_year = periods_per_year_override or periods_per_year_from_code(
        freq_summary.target
    )
    missing_payload = {
        "policy": missing_result.default_policy,
        "policy_map": missing_result.policy,
        "limit": missing_result.default_limit,
        "limit_map": missing_result.limit,
        "dropped_assets": list(missing_result.dropped_assets),
        "filled_assets": {asset: count for asset, count in missing_result.filled_cells},
        "total_filled": missing_result.total_filled,
    }

    preprocess_info = {
        "input_frequency": frequency_payload["code"],
        "input_frequency_details": frequency_payload,
        "resampled_to_monthly": normalised,
        "missing": missing_meta,
        "missing_data_policy": missing_payload,
    }
    preprocess_info["summary"] = _preprocessing_summary(
        freq_code,
        normalised=normalised,
        missing_summary=missing_meta.summary,
    )

    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    in_sdate, in_edate = _parse_month(in_start), _parse_month(in_end)
    out_sdate, out_edate = _parse_month(out_start), _parse_month(out_end)

    in_df = df[(df[date_col] >= in_sdate) & (df[date_col] <= in_edate)].set_index(
        date_col
    )
    out_df = df[(df[date_col] >= out_sdate) & (df[date_col] <= out_edate)].set_index(
        date_col
    )

    if in_df.empty or out_df.empty:
        return None

    ret_cols = [c for c in df.columns if c != date_col]
    if not ret_cols:
        return None
    if indices_list:
        idx_set = set(indices_list)  # pragma: no cover - seldom used
        ret_cols = [c for c in ret_cols if c not in idx_set]  # pragma: no cover
    else:
        indices_list = []
    rf_col = min(ret_cols, key=lambda c: df[c].std())
    fund_cols = [c for c in ret_cols if c != rf_col]

    # determine which index columns have complete data
    valid_indices: list[str] = []
    if indices_list:
        idx_in_ok = ~in_df[indices_list].isna().any()  # pragma: no cover
        idx_out_ok = ~out_df[indices_list].isna().any()  # pragma: no cover
        valid_indices = [
            c for c in indices_list if idx_in_ok[c] and idx_out_ok[c]
        ]  # pragma: no cover

    # keep only funds that satisfy missing-data policy in both windows
    # default is strict completeness; optionally allow small gaps if
    # stats_cfg carries an `na_as_zero_cfg` with tolerances.
    def _max_consecutive_nans(s: pd.Series) -> int:
        is_na = s.isna().astype(int)
        # count consecutive runs
        runs = is_na.groupby((is_na != is_na.shift()).cumsum()).cumsum() * is_na
        return int(runs.max() if not runs.empty else 0)

    na_cfg = getattr(stats_cfg, "na_as_zero_cfg", None)
    if na_cfg and bool(na_cfg.get("enabled", False)):
        max_missing = int(na_cfg.get("max_missing_per_window", 0))
        max_gap = int(na_cfg.get("max_consecutive_gap", 0))

        def _ok_window(window: pd.DataFrame, col: str) -> bool:
            s = window[col]
            missing = int(s.isna().sum())
            if missing == 0:
                return True
            if missing > max_missing:
                return False
            return _max_consecutive_nans(s) <= max_gap

        fund_cols = [
            c for c in fund_cols if _ok_window(in_df, c) and _ok_window(out_df, c)
        ]
    else:
        in_ok = ~in_df[fund_cols].isna().any()
        out_ok = ~out_df[fund_cols].isna().any()
        fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]

    if stats_cfg is None:
        stats_cfg = RiskStatsConfig(risk_free=0.0)

    if selection_mode == "random" and len(fund_cols) > random_n:
        rng = np.random.default_rng(seed)
        fund_cols = rng.choice(fund_cols, size=random_n, replace=False).tolist()
    elif selection_mode == "rank":
        mask = (df[date_col] >= in_sdate) & (df[date_col] <= in_edate)
        sub = df.loc[mask, fund_cols]
        window_key = make_window_key(in_start, in_end, fund_cols, stats_cfg)
        bundle = get_window_metric_bundle(window_key)
        rank_options: dict[str, Any] = dict(rank_kwargs or {})
        fund_cols = rank_select_funds(
            sub,
            stats_cfg,
            **rank_options,
            window_key=window_key,
            bundle=bundle,
        )
    elif selection_mode == "manual":
        if manual_funds:  # pragma: no cover - rarely hit
            fund_cols = [c for c in fund_cols if c in manual_funds]
        else:
            fund_cols = []  # pragma: no cover

    if not fund_cols:
        return None
    score_frame = single_period_run(
        df[[date_col] + fund_cols], in_start, in_end, stats_cfg=stats_cfg
    )

    weight_engine_fallback: dict[str, str] | None = None
    if (
        custom_weights is None
        and weighting_scheme
        and weighting_scheme.lower() != "equal"
    ):
        try:
            from .plugins import create_weight_engine

            cov = in_df[fund_cols].cov()
            engine = create_weight_engine(weighting_scheme.lower())
            w_series = engine.weight(cov).reindex(fund_cols).fillna(0.0)
            custom_weights = {c: float(w_series.get(c, 0.0) * 100.0) for c in fund_cols}
            logger.setLevel(logging.DEBUG)
            logger.debug("Successfully created %s weight engine", weighting_scheme)
        except Exception as e:  # pragma: no cover - exercised via tests
            msg = (
                "Weight engine '%s' failed (%s: %s); falling back to equal weights"
                % (weighting_scheme, type(e).__name__, e)
            )
            logger.warning(msg)
            logger.debug(
                "Weight engine creation failed, falling back to equal weights: %s", e
            )
            weight_engine_fallback = {
                "engine": str(weighting_scheme),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            custom_weights = None

    if custom_weights is None:
        custom_weights = {c: 100 / len(fund_cols) for c in fund_cols}

    base_series = pd.Series(
        {c: float(custom_weights.get(c, 0.0)) / 100.0 for c in fund_cols},
        dtype=float,
    )
    if float(base_series.sum()) <= 0:
        base_series = pd.Series(
            np.repeat(1.0 / len(fund_cols), len(fund_cols)),
            index=fund_cols,
            dtype=float,
        )

    constraints_cfg = constraints or {}
    if not isinstance(constraints_cfg, Mapping):
        constraints_cfg = {}
    long_only = bool(constraints_cfg.get("long_only", True))
    raw_max_weight = constraints_cfg.get("max_weight")
    try:
        max_weight_val = float(raw_max_weight) if raw_max_weight is not None else None
    except (TypeError, ValueError):
        max_weight_val = None
    raw_group_caps = constraints_cfg.get("group_caps")
    group_caps_map = (
        {str(k): float(v) for k, v in raw_group_caps.items()}
        if isinstance(raw_group_caps, Mapping)
        else None
    )
    raw_groups = constraints_cfg.get("groups")
    groups_map = (
        {str(k): str(v) for k, v in raw_groups.items()}
        if isinstance(raw_groups, Mapping)
        else None
    )

    window_cfg = dict(risk_window or {})
    try:
        window_length = int(window_cfg.get("length", len(in_df)))
    except (TypeError, ValueError):
        window_length = len(in_df)
    if window_length <= 0:
        window_length = max(len(in_df), 1)
    decay_mode = str(window_cfg.get("decay", "simple"))
    lambda_value = window_cfg.get("lambda", window_cfg.get("ewma_lambda", 0.94))
    try:
        ewma_lambda = float(lambda_value)
    except (TypeError, ValueError):
        ewma_lambda = 0.94
    window_spec = RiskWindow(
        length=window_length, decay=decay_mode, ewma_lambda=ewma_lambda
    )

    turnover_cap = None
    if max_turnover is not None:
        try:
            mt = float(max_turnover)
        except (TypeError, ValueError):
            mt = None
        if mt is not None and mt > 0:
            turnover_cap = mt

    risk_diagnostics: RiskDiagnostics

    effective_signal_spec = signal_spec or TrendSpec(
        window=window_spec.length,
        min_periods=None,
        lag=1,
        vol_adjust=False,
        vol_target=None,
        zscore=False,
    )
    signal_inputs = (
        df.set_index(date_col)[fund_cols].astype(float)
        if fund_cols
        else pd.DataFrame(dtype=float)
    )
    if not signal_inputs.empty:
        signal_frame = compute_trend_signals(signal_inputs, effective_signal_spec)
    else:
        signal_frame = pd.DataFrame(dtype=float)

    try:
        weights_series, risk_diagnostics = compute_constrained_weights(
            base_series,
            in_df[fund_cols],
            window=window_spec,
            target_vol=target_vol,
            periods_per_year=periods_per_year,
            floor_vol=min_floor if min_floor > 0 else None,
            long_only=long_only,
            max_weight=max_weight_val,
            previous_weights=previous_weights,
            max_turnover=turnover_cap,
            group_caps=group_caps_map,
            groups=groups_map,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Risk controls failed; falling back to base weights: %s", exc, exc_info=True
        )
        weights_series = base_series.copy()
        asset_vol = realised_volatility(
            in_df[fund_cols], window_spec, periods_per_year=periods_per_year
        )
        latest_vol = asset_vol.iloc[-1].reindex(fund_cols)
        latest_vol = latest_vol.ffill().bfill()
        positive = latest_vol[latest_vol > 0]
        fallback_vol = float(positive.min()) if not positive.empty else 1.0
        latest_vol = latest_vol.fillna(fallback_vol)
        if min_floor > 0:
            latest_vol = latest_vol.clip(lower=min_floor)
        scale_factors = (
            pd.Series(target_vol, index=fund_cols, dtype=float)
            .div(latest_vol)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        scaled_returns = in_df[fund_cols].mul(scale_factors, axis=1)
        portfolio_returns = scaled_returns.mul(weights_series, axis=1).sum(axis=1)
        portfolio_vol = realised_volatility(
            portfolio_returns.to_frame("portfolio"),
            window_spec,
            periods_per_year=periods_per_year,
        )["portfolio"]
        risk_diagnostics = RiskDiagnostics(
            asset_volatility=asset_vol,
            portfolio_volatility=portfolio_vol,
            turnover=pd.Series(dtype=float, name="turnover"),
            turnover_value=float("nan"),
            scale_factors=scale_factors,
        )

    weights_series = weights_series.reindex(fund_cols).fillna(0.0)
    scale_factors = risk_diagnostics.scale_factors.reindex(fund_cols).fillna(0.0)

    in_scaled = in_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    out_scaled = out_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    in_scaled = in_scaled.clip(lower=-1.0)
    out_scaled = out_scaled.clip(lower=-1.0)

    if warmup > 0:
        warmup_in = min(warmup, len(in_scaled))
        warmup_out = min(warmup, len(out_scaled))
        if warmup_in:
            in_scaled.iloc[:warmup_in] = 0.0
        if warmup_out:
            out_scaled.iloc[:warmup_out] = 0.0

    in_scaled = in_scaled.fillna(0.0)
    out_scaled = out_scaled.fillna(0.0)

    rf_in = in_df[rf_col]
    rf_out = out_df[rf_col]

    want_avg_corr = False
    try:
        reg = getattr(stats_cfg, "metrics_to_run", []) or []
        want_avg_corr = "AvgCorr" in reg
    except Exception:  # pragma: no cover - defensive
        want_avg_corr = False

    is_avg_corr: dict[str, float] | None = None
    os_avg_corr: dict[str, float] | None = None
    if want_avg_corr and len(fund_cols) > 1:
        try:
            corr_in = in_scaled[fund_cols].corr()
            corr_out = out_scaled[fund_cols].corr()
            n_f = len(fund_cols)
            is_avg_corr = {}
            os_avg_corr = {}
            denominator = float(n_f - 1) if n_f > 1 else 1.0
            for f in fund_cols:
                in_sum = cast(float, corr_in.loc[f].sum())
                out_sum = cast(float, corr_out.loc[f].sum())
                in_val = (in_sum - 1.0) / denominator
                out_val = (out_sum - 1.0) / denominator
                is_avg_corr[f] = float(in_val)
                os_avg_corr[f] = float(out_val)
        except Exception:  # pragma: no cover - defensive
            is_avg_corr = None
            os_avg_corr = None

    in_stats = _compute_stats(
        in_scaled,
        rf_in,
        in_sample_avg_corr=is_avg_corr,
        out_sample_avg_corr=None,
    )
    out_stats = _compute_stats(
        out_scaled,
        rf_out,
        in_sample_avg_corr=None,
        out_sample_avg_corr=os_avg_corr,
    )
    out_stats_raw = _compute_stats(
        out_df[fund_cols],
        rf_out,
        in_sample_avg_corr=None,
        out_sample_avg_corr=os_avg_corr,
    )

    ew_weights = np.repeat(1.0 / len(fund_cols), len(fund_cols))
    ew_w_dict = {c: w for c, w in zip(fund_cols, ew_weights)}
    in_ew = calc_portfolio_returns(ew_weights, in_scaled)
    out_ew = calc_portfolio_returns(ew_weights, out_scaled)
    out_ew_raw = calc_portfolio_returns(ew_weights, out_df[fund_cols])

    in_ew_stats = _compute_stats(pd.DataFrame({"ew": in_ew}), rf_in)["ew"]
    out_ew_stats = _compute_stats(pd.DataFrame({"ew": out_ew}), rf_out)["ew"]
    out_ew_stats_raw = _compute_stats(pd.DataFrame({"ew": out_ew_raw}), rf_out)["ew"]

    user_w = weights_series.to_numpy(dtype=float, copy=False)
    user_w_dict = {c: float(weights_series[c]) for c in fund_cols}

    in_user = calc_portfolio_returns(user_w, in_scaled)
    out_user = calc_portfolio_returns(user_w, out_scaled)
    out_user_raw = calc_portfolio_returns(user_w, out_df[fund_cols])

    in_user_stats = _compute_stats(pd.DataFrame({"user": in_user}), rf_in)["user"]
    out_user_stats = _compute_stats(pd.DataFrame({"user": out_user}), rf_out)["user"]
    out_user_stats_raw = _compute_stats(pd.DataFrame({"user": out_user_raw}), rf_out)[
        "user"
    ]

    benchmark_stats: dict[str, dict[str, _Stats]] = {}
    benchmark_ir: dict[str, dict[str, float]] = {}
    all_benchmarks: dict[str, str] = {}
    if benchmarks:
        all_benchmarks.update(benchmarks)
    for idx in valid_indices:
        if idx not in all_benchmarks:
            all_benchmarks[idx] = idx

    risk_payload = {
        "asset_volatility": risk_diagnostics.asset_volatility,
        "portfolio_volatility": risk_diagnostics.portfolio_volatility,
        "turnover": risk_diagnostics.turnover,
        "turnover_value": risk_diagnostics.turnover_value,
        "scale_factors": scale_factors,
        "final_weights": weights_series,
    }

    for label, col in all_benchmarks.items():
        if col not in in_df.columns or col not in out_df.columns:
            continue
        benchmark_stats[label] = {
            "in_sample": _compute_stats(pd.DataFrame({label: in_df[col]}), rf_in)[
                label
            ],
            "out_sample": _compute_stats(pd.DataFrame({label: out_df[col]}), rf_out)[
                label
            ],
        }
        ir_series = information_ratio(out_scaled[fund_cols], out_df[col])
        ir_dict = (
            ir_series.to_dict()
            if isinstance(ir_series, pd.Series)
            else {fund_cols[0]: float(ir_series)}
        )
        # Add portfolio-level IR references for context
        try:
            ir_eq = information_ratio(out_ew_raw, out_df[col])
            ir_usr = information_ratio(out_user_raw, out_df[col])
            # Best effort conversion; skip if not scalar convertible
            ir_dict["equal_weight"] = (
                float(ir_eq)
                if isinstance(ir_eq, (float, int, np.floating))
                else float("nan")
            )
            ir_dict["user_weight"] = (
                float(ir_usr)
                if isinstance(ir_usr, (float, int, np.floating))
                else float("nan")
            )
        except Exception:
            # Leave without portfolio-level IRs if computation fails
            pass
        benchmark_ir[label] = ir_dict

    regime_returns_map: dict[str, pd.Series] = {
        "User": out_user.astype(float, copy=False),
        "Equal-Weight": out_ew.astype(float, copy=False),
    }
    regime_payload = build_regime_payload(
        data=df,
        out_index=out_df.index,
        returns_map=regime_returns_map,
        risk_free=rf_out,
        config=regime_cfg,
        freq_code=freq_summary.target,
        periods_per_year=periods_per_year,
    )

    return {
        "selected_funds": fund_cols,
        "in_sample_scaled": in_scaled,
        "out_sample_scaled": out_scaled,
        "in_sample_stats": in_stats,
        "out_sample_stats": out_stats,
        "out_sample_stats_raw": out_stats_raw,
        "in_ew_stats": in_ew_stats,
        "out_ew_stats": out_ew_stats,
        "out_ew_stats_raw": out_ew_stats_raw,
        "in_user_stats": in_user_stats,
        "out_user_stats": out_user_stats,
        "out_user_stats_raw": out_user_stats_raw,
        "ew_weights": ew_w_dict,
        "fund_weights": user_w_dict,
        "benchmark_stats": benchmark_stats,
        "benchmark_ir": benchmark_ir,
        "score_frame": score_frame,
        "weight_engine_fallback": weight_engine_fallback,
        "preprocessing": preprocess_info,
        "preprocessing_summary": preprocess_info.get("summary"),
        "risk_diagnostics": risk_payload,
        "signal_frame": signal_frame,
        "signal_spec": effective_signal_spec,
        "performance_by_regime": regime_payload.get("table", pd.DataFrame()),
        "regime_labels": regime_payload.get("labels", pd.Series(dtype="string")),
        "regime_labels_out": regime_payload.get(
            "out_labels", pd.Series(dtype="string")
        ),
        "regime_notes": regime_payload.get("notes", []),
        "regime_settings": regime_payload.get("settings", {}),
        "regime_summary": regime_payload.get("summary"),
    }


def run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float,
    monthly_cost: float,
    *,
    floor_vol: float | None = None,
    warmup_periods: int = 0,
    selection_mode: str = "all",
    random_n: int = 8,
    custom_weights: dict[str, float] | None = None,
    rank_kwargs: Mapping[str, Any] | None = None,
    manual_funds: list[str] | None = None,
    indices_list: list[str] | None = None,
    benchmarks: dict[str, str] | None = None,
    seed: int = 42,
    stats_cfg: RiskStatsConfig | None = None,
    weighting_scheme: str | None = None,
    constraints: dict[str, Any] | None = None,
    missing_policy: str | Mapping[str, str] | None = None,
    missing_limit: int | Mapping[str, int | None] | None = None,
    risk_window: Mapping[str, Any] | None = None,
    periods_per_year: float | None = None,
    previous_weights: Mapping[str, float] | None = None,
    max_turnover: float | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
) -> dict[str, object] | None:
    """Backward-compatible wrapper around ``_run_analysis``."""
    return _run_analysis(
        df,
        in_start,
        in_end,
        out_start,
        out_end,
        target_vol,
        monthly_cost,
        floor_vol=floor_vol,
        warmup_periods=warmup_periods,
        selection_mode=selection_mode,
        random_n=random_n,
        custom_weights=custom_weights,
        rank_kwargs=rank_kwargs,
        manual_funds=manual_funds,
        indices_list=indices_list,
        benchmarks=benchmarks,
        seed=seed,
        stats_cfg=stats_cfg,
        weighting_scheme=weighting_scheme,
        constraints=constraints,
        missing_policy=missing_policy,
        missing_limit=missing_limit,
        risk_window=risk_window,
        periods_per_year_override=periods_per_year,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        signal_spec=signal_spec,
        regime_cfg=regime_cfg,
    )


def run(cfg: Config) -> pd.DataFrame:
    """Execute the analysis pipeline based on ``cfg``."""
    cfg = _unwrap_cfg(cfg)
    data_settings = _cfg_section(cfg, "data")
    csv_path = _section_get(data_settings, "csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    missing_policy_cfg = _section_get(data_settings, "missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = _section_get(data_settings, "nan_policy")
    missing_limit_cfg = _section_get(data_settings, "missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = _section_get(data_settings, "nan_limit")

    df = load_csv(
        csv_path,
        errors="raise",
        missing_policy=missing_policy_cfg,
        missing_limit=missing_limit_cfg,
    )
    df = cast(pd.DataFrame, df)

    split_cfg = _cfg_section(cfg, "sample_split")
    resolved_split = _resolve_sample_split(df, split_cfg)
    metrics_section = _cfg_section(cfg, "metrics")
    metrics_list = _section_get(metrics_section, "registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    preprocessing_section = _cfg_section(cfg, "preprocessing")
    missing_section = _section_get(preprocessing_section, "missing_data")
    if not isinstance(missing_section, Mapping):
        missing_section = None
    policy_spec, limit_spec = _policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
    )

    vol_adjust = _cfg_section(cfg, "vol_adjust")
    run_settings = _cfg_section(cfg, "run")
    portfolio_cfg = _cfg_section(cfg, "portfolio")
    trend_spec = _build_trend_spec(cfg, vol_adjust)

    res = _run_analysis(
        df,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        _section_get(vol_adjust, "target_vol", 1.0),
        _section_get(run_settings, "monthly_cost", 0.0),
        floor_vol=_section_get(vol_adjust, "floor_vol"),
        warmup_periods=int(_section_get(vol_adjust, "warmup_periods", 0) or 0),
        selection_mode=_section_get(portfolio_cfg, "selection_mode", "all"),
        random_n=_section_get(portfolio_cfg, "random_n", 8),
        custom_weights=_section_get(portfolio_cfg, "custom_weights"),
        rank_kwargs=_section_get(portfolio_cfg, "rank"),
        manual_funds=_section_get(portfolio_cfg, "manual_list"),
        indices_list=_section_get(portfolio_cfg, "indices_list"),
        benchmarks=_cfg_value(cfg, "benchmarks"),
        seed=_cfg_value(cfg, "seed", 42),
        constraints=_section_get(portfolio_cfg, "constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=_section_get(vol_adjust, "window"),
        previous_weights=_section_get(portfolio_cfg, "previous_weights"),
        max_turnover=_section_get(portfolio_cfg, "max_turnover"),
        signal_spec=trend_spec,
        regime_cfg=_cfg_section(cfg, "regime"),
    )
    if res is None:
        return pd.DataFrame()
    stats = cast(dict[str, _Stats], res["out_sample_stats"])
    df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in cast(
        dict[str, dict[str, float]], res.get("benchmark_ir", {})
    ).items():
        col = f"ir_{label}"
        df[col] = pd.Series(
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )
    return df


def run_full(cfg: Config) -> dict[str, object]:
    """Return the full analysis results based on ``cfg``."""
    cfg = _unwrap_cfg(cfg)
    data_settings = _cfg_section(cfg, "data")
    csv_path = _section_get(data_settings, "csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    missing_policy_cfg = _section_get(data_settings, "missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = _section_get(data_settings, "nan_policy")
    missing_limit_cfg = _section_get(data_settings, "missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = _section_get(data_settings, "nan_limit")

    df = load_csv(
        csv_path,
        errors="raise",
        missing_policy=missing_policy_cfg,
        missing_limit=missing_limit_cfg,
    )
    df = cast(pd.DataFrame, df)

    split_cfg = _cfg_section(cfg, "sample_split")
    resolved_split = _resolve_sample_split(df, split_cfg)
    metrics_section = _cfg_section(cfg, "metrics")
    metrics_list = _section_get(metrics_section, "registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    preprocessing_section = _cfg_section(cfg, "preprocessing")
    missing_section = _section_get(preprocessing_section, "missing_data")
    if not isinstance(missing_section, Mapping):
        missing_section = None
    policy_spec, limit_spec = _policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
    )

    vol_adjust = _cfg_section(cfg, "vol_adjust")
    run_settings = _cfg_section(cfg, "run")
    portfolio_cfg = _cfg_section(cfg, "portfolio")
    trend_spec = _build_trend_spec(cfg, vol_adjust)

    res = _run_analysis(
        df,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        _section_get(vol_adjust, "target_vol", 1.0),
        _section_get(run_settings, "monthly_cost", 0.0),
        floor_vol=_section_get(vol_adjust, "floor_vol"),
        warmup_periods=int(_section_get(vol_adjust, "warmup_periods", 0) or 0),
        selection_mode=_section_get(portfolio_cfg, "selection_mode", "all"),
        random_n=_section_get(portfolio_cfg, "random_n", 8),
        custom_weights=_section_get(portfolio_cfg, "custom_weights"),
        rank_kwargs=_section_get(portfolio_cfg, "rank"),
        manual_funds=_section_get(portfolio_cfg, "manual_list"),
        indices_list=_section_get(portfolio_cfg, "indices_list"),
        benchmarks=_cfg_value(cfg, "benchmarks"),
        seed=_cfg_value(cfg, "seed", 42),
        weighting_scheme=_section_get(portfolio_cfg, "weighting_scheme", "equal"),
        constraints=_section_get(portfolio_cfg, "constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=_section_get(vol_adjust, "window"),
        previous_weights=_section_get(portfolio_cfg, "previous_weights"),
        max_turnover=_section_get(portfolio_cfg, "max_turnover"),
        signal_spec=trend_spec,
        regime_cfg=_cfg_section(cfg, "regime"),
    )
    return {} if res is None else res


# --- Shift-safe helpers ----------------------------------------------------


def compute_signal(
    df: pd.DataFrame,
    *,
    column: str = "returns",
    window: int = 3,
    min_periods: int | None = None,
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

    cache = get_cache()

    def _compute() -> pd.Series:
        spec = TrendSpec(
            window=window,
            min_periods=effective_min_periods,
            lag=1,
            vol_adjust=False,
            zscore=False,
        )
        frame = compute_trend_signals(df[[column]].astype(float), spec)
        series = frame[column].rename(f"{column}_signal")
        return series.astype(float)

    if cache.is_enabled():
        dataset_hash = compute_dataset_hash([base])
        idx = base.index
        # Best-effort frequency tag; keep simple to satisfy type checker
        try:  # pragma: no cover - heuristic only
            freq = getattr(idx, "freq", None)
            if freq is not None:
                freq = str(freq)
            else:
                freq = None
        except Exception:  # noqa: BLE001
            freq = None
        freq_tag = freq or "unknown"
        method_tag = f"trend_spec_window{window}_min{effective_min_periods}"
        return cache.get_or_compute(
            dataset_hash, int(window), freq_tag, method_tag, _compute
        )

    return _compute()


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


# Export alias for backward compatibility
Stats = _Stats

__all__ = [
    "Stats",  # noqa: F822
    "calc_portfolio_returns",
    "single_period_run",
    "run_analysis",
    "run",
    "run_full",
    "compute_signal",
    "position_from_signal",
]


def __getattr__(name: str) -> object:
    if name == "Stats":
        return _Stats
    raise AttributeError(name)


del Stats
