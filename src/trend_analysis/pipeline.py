from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from trend.diagnostics import DiagnosticResult

from . import pipeline_helpers
from .core.rank_selection import (
    RiskStatsConfig,
    get_window_metric_bundle,
    make_window_key,
    rank_select_funds,
)
from .data import identify_risk_free_fund, load_csv
from .diagnostics import PipelineReasonCode, PipelineResult, RunPayload
from .metrics import (
    annual_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)
from .perf.rolling_cache import compute_dataset_hash, get_cache
from .pipeline_entrypoints import ConfigBindings, run_from_config, run_full_from_config
from .pipeline_helpers import (
    _attach_calendar_settings,
    _build_trend_spec,
    _cfg_section,
    _cfg_value,
    _derive_split_from_periods,
    _empty_run_full_result,
    _policy_from_config,
    _resolve_target_vol,
    _section_get,
    _unwrap_cfg,
)
from .pipeline_helpers import (
    _resolve_sample_split as _resolve_sample_split_impl,
)
from .pipeline_helpers import (
    compute_signal as _compute_signal_impl,
)
from .pipeline_helpers import (
    position_from_signal as _position_from_signal_impl,
)
from .pipeline_runner import (
    _run_analysis as _run_analysis_impl,
)
from .pipeline_runner import (
    _run_analysis_with_diagnostics as _run_analysis_with_diagnostics_impl,
)
from .portfolio import apply_weight_policy
from .regimes import build_regime_payload
from .risk import compute_constrained_weights, realised_volatility
from .signals import TrendSpec, compute_trend_signals
from .stages import portfolio as portfolio_stage
from .stages import preprocessing as preprocessing_stage
from .stages import selection as selection_stage
from .stages.portfolio import (
    _assemble_analysis_output as _assemble_analysis_output_impl,
)
from .stages.portfolio import (
    _compute_stats,
    _Stats,
    calc_portfolio_returns,
)
from .stages.portfolio import (
    _compute_weights_and_stats as _compute_weights_and_stats_impl,
)
from .stages.preprocessing import (
    _build_sample_windows as _build_sample_windows_impl,
)
from .stages.preprocessing import (
    _frequency_label,
    _preprocessing_summary,
    _WindowStage,
)
from .stages.preprocessing import (
    _prepare_input_data as _prepare_input_data_impl,
)
from .stages.preprocessing import (
    _prepare_preprocess_stage as _prepare_preprocess_stage_impl,
)
from .stages.selection import (
    _resolve_risk_free_column,
    single_period_run,
)
from .stages.selection import (
    _select_universe as _select_universe_impl,
)
from .time_utils import align_calendar
from .util.frequency import FrequencySummary, detect_frequency
from .util.missing import MissingPolicyResult, apply_missing_policy
from .weights.robust_config import weight_engine_params_from_robustness

logger = logging.getLogger(__name__)


def _sync_stage_dependencies() -> None:
    """Synchronize stage module globals with pipeline-level bindings.

    This ensures monkeypatching pipeline functions affects stage execution.
    """
    # These assignments are for runtime patching; mypy may or may not see the
    # attributes depending on module resolution. Suppress with type: ignore[attr-defined].
    setattr(preprocessing_stage, "detect_frequency", detect_frequency)
    setattr(preprocessing_stage, "apply_missing_policy", apply_missing_policy)
    setattr(preprocessing_stage, "align_calendar", align_calendar)
    setattr(preprocessing_stage, "_prepare_input_data", _prepare_input_data)

    setattr(selection_stage, "rank_select_funds", rank_select_funds)
    setattr(selection_stage, "get_window_metric_bundle", get_window_metric_bundle)
    setattr(selection_stage, "make_window_key", make_window_key)
    setattr(selection_stage, "single_period_run", single_period_run)
    setattr(selection_stage, "_resolve_risk_free_column", _resolve_risk_free_column)
    setattr(selection_stage, "identify_risk_free_fund", identify_risk_free_fund)

    setattr(portfolio_stage, "compute_trend_signals", compute_trend_signals)
    setattr(portfolio_stage, "compute_constrained_weights", compute_constrained_weights)
    setattr(portfolio_stage, "realised_volatility", realised_volatility)
    setattr(portfolio_stage, "apply_weight_policy", apply_weight_policy)
    setattr(portfolio_stage, "information_ratio", information_ratio)
    setattr(portfolio_stage, "annual_return", annual_return)
    setattr(portfolio_stage, "volatility", volatility)
    setattr(portfolio_stage, "sharpe_ratio", sharpe_ratio)
    setattr(portfolio_stage, "sortino_ratio", sortino_ratio)
    setattr(portfolio_stage, "max_drawdown", max_drawdown)
    setattr(portfolio_stage, "build_regime_payload", build_regime_payload)
    setattr(portfolio_stage, "avg_corr_handler", _avg_corr_handler)
    setattr(portfolio_stage, "calc_portfolio_returns", calc_portfolio_returns)


def _call_with_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_stage_dependencies()
    return func(*args, **kwargs)


def _avg_corr_handler(
    in_scaled: pd.DataFrame,
    out_scaled: pd.DataFrame,
    fund_cols: list[str],
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    corr_in = in_scaled[fund_cols].corr()
    corr_out = out_scaled[fund_cols].corr()
    n_f = len(fund_cols)
    is_avg_corr: dict[str, float] = {}
    os_avg_corr: dict[str, float] = {}
    denominator = float(n_f - 1) if n_f > 1 else 1.0
    for f in fund_cols:
        in_sum = float(corr_in.loc[f].sum())
        out_sum = float(corr_out.loc[f].sum())
        in_val = (in_sum - 1.0) / denominator
        out_val = (out_sum - 1.0) / denominator
        is_avg_corr[f] = float(in_val)
        os_avg_corr[f] = float(out_val)
    return is_avg_corr, os_avg_corr


def _prepare_input_data(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_prepare_input_data_impl, *args, **kwargs)


def _prepare_preprocess_stage(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_prepare_preprocess_stage_impl, *args, **kwargs)


def _build_sample_windows(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_build_sample_windows_impl, *args, **kwargs)


def _select_universe(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_select_universe_impl, *args, **kwargs)


def _compute_weights_and_stats(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_compute_weights_and_stats_impl, *args, **kwargs)


def _assemble_analysis_output(*args: Any, **kwargs: Any) -> Any:
    return _call_with_sync(_assemble_analysis_output_impl, *args, **kwargs)


def _run_analysis_with_diagnostics(*args: Any, **kwargs: Any) -> PipelineResult:
    result = _call_with_sync(_run_analysis_with_diagnostics_impl, *args, **kwargs)
    return result  # type: ignore[no-any-return]


def _run_analysis(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible wrapper returning raw payloads for tests."""
    return _call_with_sync(_run_analysis_impl, *args, **kwargs)


def _resolve_sample_split(*args: Any, **kwargs: Any) -> Any:
    pipeline_helpers._derive_split_from_periods = _derive_split_from_periods
    return _resolve_sample_split_impl(*args, **kwargs)


_DEFAULT_RUN_ANALYSIS = _run_analysis


def _invoke_analysis_with_diag(*args: Any, **kwargs: Any) -> PipelineResult:
    """Call the patched analysis hook and normalise into a PipelineResult."""

    if _run_analysis is _DEFAULT_RUN_ANALYSIS:
        return _run_analysis_with_diagnostics(*args, **kwargs)
    patched_result = _run_analysis(*args, **kwargs)
    if isinstance(patched_result, PipelineResult):
        return patched_result
    if isinstance(patched_result, DiagnosticResult):
        return PipelineResult(
            value=patched_result.value,
            diagnostic=patched_result.diagnostic,
        )
    return PipelineResult(value=patched_result, diagnostic=None)


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
    max_turnover: float | None = None,
    signal_spec: TrendSpec | None = None,
    regime_cfg: Mapping[str, Any] | None = None,
    calendar_frequency: str | None = None,
    calendar_timezone: str | None = None,
    holiday_calendar: str | None = None,
    weight_policy: Mapping[str, Any] | None = None,
    risk_free_column: str | None = None,
    allow_risk_free_fallback: bool | None = False,
    weight_engine_params: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Diagnostics-aware wrapper mirroring ``_run_analysis``."""
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
    return _invoke_analysis_with_diag(
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
        weight_engine_params=weight_engine_params,
    )


def _bindings() -> ConfigBindings:
    return ConfigBindings(
        load_csv=load_csv,
        attach_calendar_settings=_attach_calendar_settings,
        unwrap_cfg=_unwrap_cfg,
        cfg_section=_cfg_section,
        section_get=_section_get,
        cfg_value=_cfg_value,
        resolve_sample_split=_resolve_sample_split,
        policy_from_config=_policy_from_config,
        build_trend_spec=_build_trend_spec,
        resolve_target_vol=_resolve_target_vol,
        invoke_analysis_with_diag=_invoke_analysis_with_diag,
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
                "pipeline.run expected a DataFrame payload; " f"received {type(payload)!r}"
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
    return _compute_signal_impl(
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
    return _position_from_signal_impl(
        signal,
        long_position=long_position,
        short_position=short_position,
        neutral_position=neutral_position,
    )


# Export alias for backward compatibility
Stats = _Stats

__all__ = [
    "FrequencySummary",
    "MissingPolicyResult",
    "PipelineReasonCode",
    "Stats",  # noqa: F822
    "_Stats",  # Direct export for type checking
    "_WindowStage",
    "_assemble_analysis_output",
    "_build_sample_windows",
    "_build_trend_spec",
    "_cfg_section",
    "_cfg_value",
    "_compute_stats",
    "_compute_weights_and_stats",
    "_derive_split_from_periods",
    "_empty_run_full_result",
    "_frequency_label",
    "_invoke_analysis_with_diag",
    "_policy_from_config",
    "_prepare_input_data",
    "_prepare_preprocess_stage",
    "_preprocessing_summary",
    "_resolve_risk_free_column",
    "_resolve_sample_split",
    "_resolve_target_vol",
    "_run_analysis",
    "_run_analysis_with_diagnostics",
    "_section_get",
    "_select_universe",
    "_unwrap_cfg",
    "calc_portfolio_returns",
    "compute_signal",
    "position_from_signal",
    "run",
    "run_analysis",
    "run_full",
    "single_period_run",
]


def __getattr__(name: str) -> object:
    if name == "Stats":
        return _Stats
    raise AttributeError(name)


del Stats
