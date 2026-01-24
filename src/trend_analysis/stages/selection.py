from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..core.rank_selection import (
    RiskStatsConfig,
    get_window_metric_bundle,
    make_window_key,
    rank_select_funds,
)
from ..data import identify_risk_free_fund
from ..diagnostics import PipelineReasonCode, PipelineResult, pipeline_failure
from ..timefreq import MONTHLY_DATE_FREQ
from .preprocessing import _PreprocessStage, _WindowStage

logger = logging.getLogger("trend_analysis.pipeline")

# Require a minimum share of coverage within the requested windows before a
# column is eligible for risk-free fallback selection. This prevents sparse
# series from being chosen and later propagating NaNs when aligned to the
# analysis windows.
_MIN_FALLBACK_COVERAGE_RATIO = 0.6

__all__ = [
    "_SelectionStage",
    "_resolve_risk_free_column",
    "_select_universe",
    "single_period_run",
]


@dataclass(slots=True)
class _SelectionStage:
    fund_cols: list[str]
    rf_col: str
    rf_source: str
    score_frame: pd.DataFrame
    risk_free_override: pd.Series
    indices_list: list[str]
    requested_indices: list[str]
    missing_indices: list[str]


def _resolve_risk_free_column(
    df: pd.DataFrame,
    *,
    date_col: str,
    indices_list: list[str] | None,
    risk_free_column: str | None,
    allow_risk_free_fallback: bool | None,
    fallback_window: pd.DataFrame | None = None,
) -> tuple[str, list[str], str]:
    """Select the risk-free column and investable funds.

    Returns
    -------
    tuple[str, list[str], str]
        ``(risk_free_column, fund_columns, source)`` where ``source`` is
        ``"configured"`` when explicitly provided or ``"fallback"`` when the
        column is inferred. When ``fallback_window`` is supplied, the
        candidate scan is restricted to that window to avoid picking columns
        that lack data for the requested analysis period.
    """

    idx_set = {str(c) for c in indices_list or []}
    candidate_df = fallback_window if fallback_window is not None else df
    if date_col not in candidate_df.columns:
        candidate_df = candidate_df.copy()
        candidate_df[date_col] = candidate_df.index

    candidate_df = candidate_df.copy()
    candidate_df[date_col] = pd.to_datetime(candidate_df[date_col])
    candidate_df.sort_values(date_col, inplace=True)

    configured_rf = (risk_free_column or "").strip()

    numeric_cols = [c for c in candidate_df.select_dtypes("number").columns if c != date_col]
    if not numeric_cols:
        raise ValueError(
            "No numeric return columns were found in the requested window; cannot select risk-free series"
        )

    expanded_df = candidate_df.set_index(date_col)
    if not expanded_df.index.is_monotonic_increasing:
        expanded_df = expanded_df.sort_index()

    # Align observed dates to calendar month-end without dropping values.
    # This handles inputs keyed on business month-end or daily observations.
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(expanded_df.index))
        # Map observations to month-end dates at midnight so they align with
        # `pd.date_range(..., freq=MONTHLY_DATE_FREQ)`.
        idx_me = idx.to_period("M").to_timestamp(how="end").normalize()
        expanded_df.index = idx_me
        if expanded_df.index.has_duplicates:
            expanded_df = expanded_df.groupby(level=0).last()
    except Exception:  # pragma: no cover - best-effort alignment
        pass

    if not expanded_df.empty:
        start_me = pd.Timestamp(expanded_df.index.min())
        end_me = pd.Timestamp(expanded_df.index.max())
        # If the risk-free series is explicitly configured, compute coverage
        # over the span where that series actually has observations. This avoids
        # penalising coverage for trailing empty months that can be introduced
        # by window reindexing elsewhere in the pipeline.
        if configured_rf and configured_rf in expanded_df.columns:
            observed_idx = expanded_df.index[expanded_df[configured_rf].notna()]
            if len(observed_idx) > 0:
                start_me = pd.Timestamp(observed_idx.min())
                end_me = pd.Timestamp(observed_idx.max())

        full_index = pd.date_range(start=start_me, end=end_me, freq=MONTHLY_DATE_FREQ)
        expanded_df = expanded_df.reindex(full_index)
    expanded_df.index.name = date_col

    total_rows = len(expanded_df)
    if total_rows == 0:
        raise ValueError("Requested window is empty; cannot select risk-free series")

    # Coverage threshold for selecting a risk-free proxy. Clamp to the window
    # length so truncated/short windows (e.g., a 1-month final OOS period)
    # don't fail with an impossible requirement like 2/1 observations.
    min_non_null = math.ceil(total_rows * _MIN_FALLBACK_COVERAGE_RATIO)
    min_non_null = max(1, min(total_rows, min_non_null))
    coverage_counts = expanded_df[numeric_cols].notna().sum()
    coverage_mask = coverage_counts >= min_non_null

    if configured_rf:
        if configured_rf == date_col:
            raise ValueError("Risk-free column cannot reuse the date column")
        if configured_rf not in df.columns:
            raise ValueError(
                f"Configured risk-free column '{configured_rf}' was not found in the dataset"
            )
        if configured_rf not in candidate_df.select_dtypes("number").columns:
            raise ValueError(f"Configured risk-free column '{configured_rf}' must be numeric")
        if configured_rf in idx_set:
            raise ValueError(
                f"Risk-free column '{configured_rf}' cannot also be listed as an index/benchmark"
            )

        # Fund candidates come from all numeric columns (excluding indices and rf).
        ret_cols = [c for c in numeric_cols if c not in idx_set]

        configured_coverage = int(coverage_counts.get(configured_rf, 0))
        if configured_coverage == 0:
            # Window may be out of data range (common in multi-period schedules).
            # Let the pipeline proceed so the period can be skipped with a
            # diagnostic instead of raising here.
            logger.warning(
                "Configured risk-free column '%s' has no coverage in the requested window; proceeding",
                configured_rf,
            )
        elif configured_coverage < min_non_null:
            raise ValueError(
                (
                    f"Configured risk-free column '{configured_rf}' has insufficient coverage "
                    f"in the requested window ({configured_coverage}/{total_rows} non-null; "
                    f"require at least {min_non_null})"
                )
            )
        rf_col = configured_rf
        source = "configured"
    else:
        # Restrict candidates to columns with sufficient non-null coverage within
        # the requested windows. This keeps the fallback selection aligned with the
        # analysis slice rather than the full dataset.
        ret_cols = [c for c in numeric_cols if c not in idx_set and coverage_mask.get(c, False)]

        if not ret_cols:
            raise ValueError(
                (
                    "No numeric return columns meet the coverage requirement "
                    f"({min_non_null}/{total_rows} non-null observations) in the requested window"
                )
            )

        # Fallback remains opt-in; only enable when explicitly requested.
        fallback_enabled = allow_risk_free_fallback is True
        if not fallback_enabled:
            raise ValueError(
                "Set data.risk_free_column or enable data.allow_risk_free_fallback to select a risk-free series."
            )
        window_df = expanded_df.reset_index().rename(columns={"index": date_col})
        probe_cols = [date_col, *ret_cols] if date_col in window_df.columns else ret_cols

        # With <2 observations, volatility is undefined (std = NaN), which can
        # cause the fallback heuristic to return NaN. Prefer obvious RF-like
        # columns or fall back deterministically.
        if total_rows < 2:
            rf_like = (
                "RF",
                "RISK_FREE",
                "RISK-FREE",
                "CASH",
                "TBILL",
                "TBILLS",
                "T-BILL",
                "T-BILLS",
            )
            by_upper = {str(c).upper(): str(c) for c in ret_cols}
            pick = None
            for key in rf_like:
                if key in by_upper:
                    pick = by_upper[key]
                    break
            rf_col = pick or sorted(map(str, ret_cols))[0]
            source = "fallback"
            logger.info(
                "Using '%s' as risk-free (fallback short-window)",
                rf_col,
            )
        else:
            detected = identify_risk_free_fund(window_df[probe_cols])
            if detected is None or (isinstance(detected, float) and math.isnan(detected)):
                raise ValueError(
                    "Risk-free fallback could not find a numeric return series in the requested window"
                )
            rf_col = str(detected)
            source = "fallback"
            logger.info(
                "Using lowest-volatility column '%s' as risk-free (fallback enabled)",
                rf_col,
            )

    fund_cols = [c for c in ret_cols if c != rf_col]
    return rf_col, fund_cols, source


def single_period_run(
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    stats_cfg: RiskStatsConfig | None = None,
    risk_free: float | pd.Series | None = None,
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
    from ..core.rank_selection import RiskStatsConfig, _compute_metric_series

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

    if window.empty:
        raise ValueError(
            "single_period_run found no rows in the requested period"
            f" {start} to {end}; check the date filter or input data."
        )

    window_no_all_nan = window.dropna(axis=1, how="all")
    if window_no_all_nan.empty:
        raise ValueError(
            "single_period_run found only empty return columns in the requested"
            f" period {start} to {end}; verify the input contains non-NaN returns."
        )

    metrics = stats_cfg.metrics_to_run
    if not metrics:
        raise ValueError("stats_cfg.metrics_to_run must not be empty")

    parts = [
        _compute_metric_series(window_no_all_nan, m, stats_cfg, risk_free_override=risk_free)
        for m in metrics
    ]
    score_frame = pd.concat(parts, axis=1)
    score_frame.columns = metrics
    score_frame.attrs["insample_len"] = len(window)
    score_frame.attrs["period"] = (start, end)
    # Optional derived correlation metric (opt-in via stats_cfg.extra_metrics)
    extra = getattr(stats_cfg, "extra_metrics", [])
    if (
        "AvgCorr" in extra
        and score_frame.shape[1] > 0
        and window_no_all_nan.shape[1] > 1
        and "AvgCorr" not in score_frame.columns
    ):
        from ..core.rank_selection import compute_metric_series_with_cache

        try:
            avg_corr_series = compute_metric_series_with_cache(
                window_no_all_nan,
                "AvgCorr",
                stats_cfg,
                risk_free_override=risk_free,
                enable_cache=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            msg = (
                "Failed to compute AvgCorr for single_period_run" f" window {start} to {end}: {exc}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from exc

        score_frame = pd.concat([score_frame, avg_corr_series], axis=1)
    return score_frame.astype(float)


def _select_universe(
    preprocess: _PreprocessStage,
    window: _WindowStage,
    *,
    in_label: str,
    in_end_label: str,
    selection_mode: str,
    random_n: int,
    custom_weights: dict[str, float] | None,
    rank_kwargs: Mapping[str, Any] | None,
    manual_funds: list[str] | None,
    indices_list: list[str] | None,
    seed: int,
    stats_cfg: RiskStatsConfig | None,
    risk_free_column: str | None,
    allow_risk_free_fallback: bool | None,
) -> _SelectionStage | PipelineResult:
    requested_indices = [str(idx) for idx in indices_list] if indices_list else []
    valid_indices: list[str] = []
    missing_indices: list[str] = []
    if requested_indices:
        available_cols = set(window.in_df.columns)
        available_indices = [idx for idx in requested_indices if idx in available_cols]
        missing_indices = [idx for idx in requested_indices if idx not in available_cols]

        for idx in available_indices:
            has_data = window.in_df[idx].notnull().any()
            if has_data:
                valid_indices.append(idx)
            else:
                missing_indices.append(idx)

        if not valid_indices:
            return pipeline_failure(
                PipelineReasonCode.INDICES_ABSENT,
                context={
                    "requested_indices": requested_indices,
                    "missing_indices": missing_indices,
                    "available_columns": sorted(available_cols),
                },
            )
    indices_list = valid_indices

    rf_col: str
    fund_cols: list[str]
    rf_source: str
    fallback_window = pd.concat(
        [window.in_df.reset_index(), window.out_df.reset_index()],
        ignore_index=True,
    )
    if preprocess.date_col not in fallback_window.columns and "index" in fallback_window.columns:
        fallback_window = fallback_window.rename(columns={"index": preprocess.date_col})
    rf_col, fund_cols, rf_source = _resolve_risk_free_column(
        preprocess.df,
        date_col=preprocess.date_col,
        indices_list=indices_list,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        fallback_window=fallback_window,
    )

    if selection_mode == "all" and custom_weights is not None:
        fund_cols = [c for c in fund_cols if c in custom_weights]
        if not fund_cols and custom_weights:
            fund_cols = list(custom_weights.keys())
        else:
            custom_weights = None

    # keep only funds that satisfy missing-data policy in both windows. The
    # default behaviour enforces strict completeness, while ``na_as_zero`` can
    # provide tolerances for total and consecutive gaps (Issue #3633).
    def _max_consecutive_nans(series: pd.Series) -> int:
        if not series.isna().any():
            return 0
        is_na = series.isna().astype(int)
        groups = (is_na != is_na.shift()).cumsum()
        runs = is_na.groupby(groups).cumsum() * is_na
        return int(runs.max()) if not runs.empty else 0

    na_cfg = getattr(stats_cfg, "na_as_zero_cfg", None) if stats_cfg else None
    if na_cfg and bool(na_cfg.get("enabled", False)):
        max_missing = int(na_cfg.get("max_missing_per_window", 0) or 0)
        max_gap = int(na_cfg.get("max_consecutive_gap", 0) or 0)

        def _window_ok(window_df: pd.DataFrame, column: str) -> bool:
            series = window_df[column]
            missing = int(series.isna().sum())
            if missing == 0:
                return True
            if missing > max_missing:
                return False
            if max_gap <= 0:
                return True
            return _max_consecutive_nans(series) <= max_gap

        fund_cols = [
            col
            for col in fund_cols
            if _window_ok(window.in_df, col) and _window_ok(window.out_df, col)
        ]
    else:
        in_ok = ~window.in_df[fund_cols].isna().any()
        out_ok = ~window.out_df[fund_cols].isna().any()
        fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]

    if stats_cfg is None:
        stats_cfg = RiskStatsConfig(risk_free=0.0)

    risk_free_override = window.in_df[rf_col]

    if selection_mode == "random" and len(fund_cols) > random_n:
        rng = np.random.default_rng(seed)
        fund_cols = rng.choice(fund_cols, size=random_n, replace=False).tolist()
    elif selection_mode == "rank":
        mask = (preprocess.df[preprocess.date_col] >= window.in_start) & (
            preprocess.df[preprocess.date_col] <= window.in_end
        )
        sub = preprocess.df.loc[mask, fund_cols]
        window_key = None
        bundle = None
        if stats_cfg is not None and fund_cols:
            try:
                window_key = make_window_key(window.in_start, window.in_end, sub.columns, stats_cfg)
            except Exception:  # pragma: no cover - defensive
                window_key = None
        if window_key is not None:
            bundle = get_window_metric_bundle(window_key)
        rank_options: dict[str, Any] = dict(rank_kwargs or {})
        rank_options.setdefault("window_key", window_key)
        rank_options.setdefault("bundle", bundle)
        rank_result = rank_select_funds(
            sub,
            stats_cfg,
            **rank_options,
            risk_free=risk_free_override,
        )
        if isinstance(rank_result, tuple):
            fund_cols = rank_result[0]
        else:
            fund_cols = rank_result
    elif selection_mode == "manual":
        if manual_funds:
            # Manual mode: use the explicitly requested funds. The caller has
            # already determined these are the target holdings, so bypass the
            # missing-data filter applied above. Only require that the column
            # exists in both in-sample and out-sample windows. This ensures
            # newly-hired funds are not filtered out due to missing data.
            available = set(window.in_df.columns) & set(window.out_df.columns)
            fund_cols = [c for c in manual_funds if c in available]
        else:
            fund_cols = []  # pragma: no cover

    if not fund_cols:
        return pipeline_failure(
            PipelineReasonCode.NO_FUNDS_SELECTED,
            context={
                "selection_mode": selection_mode,
                "universe_size": len(preprocess.value_cols_all),
            },
        )

    score_frame = single_period_run(
        preprocess.df[[preprocess.date_col] + fund_cols],
        in_label,
        in_end_label,
        stats_cfg=stats_cfg,
        risk_free=risk_free_override,
    )

    return _SelectionStage(
        fund_cols=fund_cols,
        rf_col=rf_col,
        rf_source=rf_source,
        score_frame=score_frame,
        risk_free_override=risk_free_override,
        indices_list=valid_indices,
        requested_indices=requested_indices,
        missing_indices=missing_indices,
    )
