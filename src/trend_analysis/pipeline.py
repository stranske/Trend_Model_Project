from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from .core.rank_selection import RiskStatsConfig
from .data import load_csv
from .diagnostics import PipelineReasonCode, PipelineResult, RunPayload
from .perf.rolling_cache import compute_dataset_hash, get_cache
from .pipeline_entrypoints import ConfigBindings, run_from_config, run_full_from_config
from . import pipeline_runner
from . import pipeline_helpers
from .signals import TrendSpec
from .weights.robust_config import weight_engine_params_from_robustness

logger = logging.getLogger(__name__)


def run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float | None,
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
    lambda_tc: float | None = None,
    max_turnover: float | Mapping[str, float] | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
    calendar_frequency: str | None = None,
    calendar_timezone: str | None = None,
    holiday_calendar: str | None = None,
    weight_policy: Mapping[str, Any] | None = None,
    risk_free_column: str | None = None,
    allow_risk_free_fallback: bool | None = False,
    risk_free_override: float | pd.Series | None = None,
    weight_engine_params: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Run one analysis and return its diagnostics-aware result."""
    if any(
        value is not None for value in (calendar_frequency, calendar_timezone, holiday_calendar)
    ):
        df = df.copy()
        calendar_settings = dict(getattr(df, "attrs", {}).get("calendar_settings", {}))
        if calendar_frequency is not None:
            calendar_settings["frequency"] = calendar_frequency
        if calendar_timezone is not None:
            calendar_settings["timezone"] = calendar_timezone
        if holiday_calendar is not None:
            calendar_settings["holiday_calendar"] = holiday_calendar
        df.attrs["calendar_settings"] = calendar_settings
    return pipeline_runner._run_analysis_with_diagnostics(
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
        lambda_tc=lambda_tc,
        max_turnover=max_turnover,
        signal_spec=signal_spec,
        regime_cfg=regime_cfg,
        weight_policy=weight_policy,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        risk_free_override=risk_free_override,
        weight_engine_params=weight_engine_params,
    )


def _bindings() -> ConfigBindings:
    return ConfigBindings(
        load_csv=load_csv,
        attach_calendar_settings=pipeline_helpers._attach_calendar_settings,
        cfg_section=pipeline_helpers._cfg_section,
        section_get=pipeline_helpers._section_get,
        cfg_value=pipeline_helpers._cfg_value,
        resolve_sample_split=pipeline_helpers._resolve_sample_split,
        policy_from_config=pipeline_helpers._policy_from_config,
        build_trend_spec=pipeline_helpers._build_trend_spec,
        resolve_target_vol=pipeline_helpers._resolve_target_vol,
        invoke_analysis_with_diag=pipeline_runner._run_analysis_with_diagnostics,
        weight_engine_params_from_robustness=weight_engine_params_from_robustness,
        RiskStatsConfig=RiskStatsConfig,
    )


def run(cfg: Any) -> pd.DataFrame:
    """Run the analysis pipeline and return out-of-sample metrics.

    Args:
        cfg: Config instance or mapping compatible with `Config`.

    Returns:
        DataFrame of out-of-sample metrics. If diagnostics indicate an abort,
        returns an empty DataFrame with the diagnostic attached to `attrs`.
    """
    result = run_from_config(cfg, bindings=_bindings())
    if isinstance(result, RunPayload):
        payload = result.value
        if payload is None:
            empty = pd.DataFrame()
            if result.diagnostic is not None:
                empty.attrs["diagnostic"] = result.diagnostic
            return empty
        if not isinstance(payload, pd.DataFrame):
            raise TypeError(
                f"pipeline.run expected a DataFrame payload; received {type(payload)!r}"
            )
        if result.diagnostic is not None:
            payload.attrs["diagnostic"] = result.diagnostic
        return payload
    return result


def run_full(cfg: Any) -> PipelineResult:
    """Run the analysis pipeline and return the full diagnostics payload.

    Args:
        cfg: Config instance or mapping compatible with `Config`.

    Returns:
        PipelineResult containing the payload, diagnostic info, and optional
        metadata if provided by the underlying analysis call.
    """
    return run_full_from_config(cfg, bindings=_bindings())


# --- Shift-safe helpers ----------------------------------------------------


def compute_signal(
    df: pd.DataFrame,
    *,
    column: str = "returns",
    window: int = 3,
    min_periods: int | None = None,
) -> pd.Series:
    return pipeline_helpers.compute_signal(
        df,
        column=column,
        window=window,
        min_periods=min_periods,
        get_cache_func=get_cache,
        compute_dataset_hash_func=compute_dataset_hash,
        log=logger,
    )


def position_from_signal(
    signal: pd.Series,
    *,
    long_position: float = 1.0,
    short_position: float = -1.0,
    neutral_position: float = 0.0,
) -> pd.Series:
    return pipeline_helpers.position_from_signal(
        signal,
        long_position=long_position,
        short_position=short_position,
        neutral_position=neutral_position,
    )


__all__ = [
    "PipelineReasonCode",
    "compute_signal",
    "position_from_signal",
    "run",
    "run_analysis",
    "run_full",
]
