from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .core.rank_selection import RiskStatsConfig
from .diagnostics import AnalysisResult, PipelineResult
from .pipeline_helpers import (
    _apply_regime_overrides,
    _apply_regime_weight_overrides,
    _resolve_regime_label,
)
from .signals import TrendSpec
from .stages import portfolio as portfolio_stage
from .stages import preprocessing as preprocessing_stage
from .stages import selection as selection_stage

__all__ = ["_run_analysis", "_run_analysis_with_diagnostics"]


def _run_analysis_with_diagnostics(
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
    periods_per_year_override: float | None = None,
    previous_weights: Mapping[str, float] | None = None,
    lambda_tc: float | None = None,
    max_turnover: float | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
    weight_policy: Mapping[str, Any] | None = None,
    risk_free_column: str | None = None,
    allow_risk_free_fallback: bool | None = False,
    weight_engine_params: Mapping[str, Any] | None = None,
) -> PipelineResult:
    preprocess_stage = preprocessing_stage._prepare_preprocess_stage(
        df,
        floor_vol=floor_vol,
        warmup_periods=warmup_periods,
        missing_policy=missing_policy,
        missing_limit=missing_limit,
        stats_cfg=stats_cfg,
        periods_per_year_override=periods_per_year_override,
        allow_risk_free_fallback=allow_risk_free_fallback,
    )
    if isinstance(preprocess_stage, PipelineResult):
        return preprocess_stage

    window_stage = preprocessing_stage._build_sample_windows(
        preprocess_stage,
        in_start=in_start,
        in_end=in_end,
        out_start=out_start,
        out_end=out_end,
    )
    if isinstance(window_stage, PipelineResult):
        return window_stage

    regime_label, regime_settings = _resolve_regime_label(
        preprocess_stage,
        window_stage,
        regime_cfg,
        benchmarks=benchmarks,
    )
    random_n, rank_kwargs = _apply_regime_overrides(
        random_n=random_n,
        rank_kwargs=rank_kwargs,
        regime_label=regime_label,
        settings=regime_settings,
        regime_cfg=regime_cfg,
    )
    target_vol, constraints = _apply_regime_weight_overrides(
        target_vol=target_vol,
        constraints=constraints,
        regime_label=regime_label,
        settings=regime_settings,
        regime_cfg=regime_cfg,
    )

    selection_stage_result = selection_stage._select_universe(
        preprocess_stage,
        window_stage,
        in_label=in_start,
        in_end_label=in_end,
        selection_mode=selection_mode,
        random_n=random_n,
        custom_weights=custom_weights,
        rank_kwargs=rank_kwargs,
        manual_funds=manual_funds,
        indices_list=indices_list,
        seed=seed,
        stats_cfg=stats_cfg,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
    )
    if isinstance(selection_stage_result, PipelineResult):
        return selection_stage_result

    stats_cfg_obj = stats_cfg or RiskStatsConfig(risk_free=0.0)
    computation_stage = portfolio_stage._compute_weights_and_stats(
        preprocess_stage,
        window_stage,
        selection_stage_result,
        target_vol=target_vol,
        monthly_cost=monthly_cost,
        custom_weights=custom_weights,
        weighting_scheme=weighting_scheme,
        constraints=constraints,
        risk_window=risk_window,
        previous_weights=previous_weights,
        lambda_tc=lambda_tc,
        max_turnover=max_turnover,
        signal_spec=signal_spec,
        weight_policy=weight_policy,
        warmup=preprocess_stage.warmup,
        min_floor=preprocess_stage.min_floor,
        stats_cfg=stats_cfg_obj,
        weight_engine_params=weight_engine_params,
    )

    return portfolio_stage._assemble_analysis_output(
        preprocess_stage,
        window_stage,
        selection_stage_result,
        computation_stage,
        benchmarks=benchmarks,
        regime_cfg=regime_cfg,
        target_vol=target_vol,
        monthly_cost=monthly_cost,
        min_floor=preprocess_stage.min_floor,
    )


def _run_analysis(
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
    periods_per_year_override: float | None = None,
    previous_weights: Mapping[str, float] | None = None,
    lambda_tc: float | None = None,
    max_turnover: float | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
    weight_policy: Mapping[str, Any] | None = None,
    risk_free_column: str | None = None,
    allow_risk_free_fallback: bool | None = False,
    weight_engine_params: Mapping[str, Any] | None = None,
) -> AnalysisResult | None:
    """Backward-compatible wrapper returning raw payloads for tests."""

    result = _run_analysis_with_diagnostics(
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
        periods_per_year_override=periods_per_year_override,
        previous_weights=previous_weights,
        lambda_tc=lambda_tc,
        max_turnover=max_turnover,
        signal_spec=signal_spec,
        regime_cfg=regime_cfg,
        weight_policy=weight_policy,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        weight_engine_params=weight_engine_params,
    )
    return result.unwrap()
