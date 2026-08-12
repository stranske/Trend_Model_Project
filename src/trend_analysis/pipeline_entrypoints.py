from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import pandas as pd

from trend.diagnostics import DiagnosticResult

from .diagnostics import PipelineResult, coerce_pipeline_result
from .util.risk_free import resolve_risk_free_settings

logger = logging.getLogger("trend_analysis.pipeline")


@dataclass(frozen=True, slots=True)
class ConfigBindings:
    load_csv: Any
    attach_calendar_settings: Any
    unwrap_cfg: Any
    cfg_section: Any
    section_get: Any
    cfg_value: Any
    resolve_sample_split: Any
    policy_from_config: Any
    build_trend_spec: Any
    resolve_target_vol: Any
    invoke_analysis_with_diag: Any
    weight_engine_params_from_robustness: Any
    RiskStatsConfig: Any


def _accepts_keyword(func: Any, keyword: str) -> bool:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return keyword in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


def _resolve_single_period_weighting_scheme(portfolio_cfg: Any, section_get: Any) -> str:
    """Use the same public weighting-key precedence as the multi-period engine."""

    weighting_cfg = section_get(portfolio_cfg, "weighting", {})
    nested_name = section_get(weighting_cfg, "name") if isinstance(weighting_cfg, Mapping) else None
    legacy_name = section_get(portfolio_cfg, "weighting_scheme")
    if legacy_name not in (None, "", "equal", "custom"):
        return _normalise_single_period_weighting_name(legacy_name)
    if nested_name not in (None, ""):
        return _normalise_single_period_weighting_name(nested_name)
    return _normalise_single_period_weighting_name(legacy_name)


def _normalise_single_period_weighting_name(value: Any) -> str:
    return {"ew": "equal", "robust": "robust_mv"}.get(
        str(value or "equal").strip().lower(), str(value or "equal").strip().lower()
    )


def _optional_cost_bps(value: Any, *, field: str) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"portfolio.{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"portfolio.{field} must be finite and non-negative")
    return parsed


def _resolve_single_period_monthly_cost(portfolio_cfg: Any, run_cfg: Any) -> float:
    """Translate the canonical portfolio cost model for the single-period path."""

    portfolio = dict(portfolio_cfg) if isinstance(portfolio_cfg, Mapping) else {}
    run = dict(run_cfg) if isinstance(run_cfg, Mapping) else {}
    cost_model = portfolio.get("cost_model")
    cost_model = cost_model if isinstance(cost_model, Mapping) else {}
    tc_bps = _optional_cost_bps(
        cost_model.get("per_trade_bps", cost_model.get("bps_per_trade")),
        field="cost_model.per_trade_bps",
    )
    if tc_bps is None:
        tc_bps = _optional_cost_bps(
            portfolio.get("transaction_cost_bps", 0.0), field="transaction_cost_bps"
        )
    slippage_bps = _optional_cost_bps(
        cost_model.get("half_spread_bps", cost_model.get("slippage_bps")),
        field="cost_model.half_spread_bps",
    )
    if slippage_bps is None:
        slippage_bps = _optional_cost_bps(portfolio.get("slippage_bps", 0.0), field="slippage_bps")
    if any(key in portfolio for key in ("cost_model", "transaction_cost_bps", "slippage_bps")):
        return (float(tc_bps or 0.0) + float(slippage_bps or 0.0)) / 10000.0
    try:
        monthly_cost = float(run.get("monthly_cost", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("run.monthly_cost must be numeric") from exc
    if not math.isfinite(monthly_cost) or monthly_cost < 0:
        raise ValueError("run.monthly_cost must be finite and non-negative")
    return monthly_cost


def run_from_config(cfg: Any, *, bindings: ConfigBindings) -> pd.DataFrame:
    """Run the analysis pipeline using a config-like object.

    Args:
        cfg: Config instance or mapping compatible with `Config`.
        bindings: Helper bindings that load data, resolve config sections, and
            invoke the analysis core.

    Returns:
        A DataFrame of out-of-sample metrics. When the run aborts, returns an
        empty DataFrame and attaches any diagnostics on `DataFrame.attrs`.
    """
    cfg = bindings.unwrap_cfg(cfg)
    preprocessing_section = bindings.cfg_section(cfg, "preprocessing")
    data_settings = bindings.cfg_section(cfg, "data")
    csv_path = bindings.section_get(data_settings, "csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    missing_policy_cfg = bindings.section_get(data_settings, "missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = bindings.section_get(data_settings, "nan_policy")
    missing_limit_cfg = bindings.section_get(data_settings, "missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = bindings.section_get(data_settings, "nan_limit")
    date_column_cfg = bindings.section_get(data_settings, "date_column")
    if date_column_cfg is None:
        date_column_cfg = "Date"

    load_kwargs: dict[str, Any] = {
        "errors": "raise",
        "missing_policy": missing_policy_cfg,
        "missing_limit": missing_limit_cfg,
    }
    if _accepts_keyword(bindings.load_csv, "date_column"):
        load_kwargs["date_column"] = date_column_cfg
    df = bindings.load_csv(csv_path, **load_kwargs)
    df = cast(pd.DataFrame, df)

    bindings.attach_calendar_settings(df, cfg)

    split_cfg = bindings.cfg_section(cfg, "sample_split")
    resolved_split = bindings.resolve_sample_split(df, split_cfg)
    metrics_section = bindings.cfg_section(cfg, "metrics")
    metrics_list = bindings.section_get(metrics_section, "registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import canonical_metric_list

        stats_cfg = bindings.RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    missing_section = bindings.section_get(preprocessing_section, "missing_data")
    if not isinstance(missing_section, Mapping):
        missing_section = None
    policy_spec, limit_spec = bindings.policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
    )

    vol_adjust = bindings.cfg_section(cfg, "vol_adjust")
    run_settings = bindings.cfg_section(cfg, "run")
    portfolio_cfg = bindings.cfg_section(cfg, "portfolio")
    weighting_scheme = _resolve_single_period_weighting_scheme(portfolio_cfg, bindings.section_get)
    robustness_cfg = bindings.section_get(portfolio_cfg, "robustness")
    if not isinstance(robustness_cfg, Mapping):
        robustness_cfg = bindings.cfg_section(cfg, "robustness")
    weight_engine_params = bindings.weight_engine_params_from_robustness(
        weighting_scheme, robustness_cfg
    )
    trend_spec = bindings.build_trend_spec(cfg, vol_adjust)
    lambda_tc_val = bindings.section_get(portfolio_cfg, "lambda_tc", 0.0)
    risk_free_column, allow_risk_free_fallback = resolve_risk_free_settings(
        data_settings if isinstance(data_settings, Mapping) else None
    )

    diag_res = bindings.invoke_analysis_with_diag(
        df,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        bindings.resolve_target_vol(vol_adjust),
        _resolve_single_period_monthly_cost(portfolio_cfg, run_settings),
        floor_vol=bindings.section_get(vol_adjust, "floor_vol"),
        warmup_periods=int(bindings.section_get(vol_adjust, "warmup_periods", 0) or 0),
        selection_mode=bindings.section_get(portfolio_cfg, "selection_mode", "all"),
        random_n=bindings.section_get(portfolio_cfg, "random_n", 8),
        custom_weights=bindings.section_get(portfolio_cfg, "custom_weights"),
        rank_kwargs=bindings.section_get(portfolio_cfg, "rank"),
        manual_funds=bindings.section_get(portfolio_cfg, "manual_list"),
        indices_list=bindings.section_get(portfolio_cfg, "indices_list"),
        benchmarks=bindings.cfg_value(cfg, "benchmarks"),
        seed=bindings.cfg_value(cfg, "seed", 42),
        weighting_scheme=weighting_scheme,
        constraints=bindings.section_get(portfolio_cfg, "constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=bindings.section_get(vol_adjust, "window"),
        previous_weights=bindings.section_get(portfolio_cfg, "previous_weights"),
        lambda_tc=lambda_tc_val,
        max_turnover=bindings.section_get(portfolio_cfg, "max_turnover"),
        signal_spec=trend_spec,
        regime_cfg=bindings.cfg_section(cfg, "regime"),
        weight_policy=bindings.section_get(portfolio_cfg, "weight_policy"),
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        weight_engine_params=weight_engine_params,
    )
    diag = diag_res.diagnostic
    if diag_res.value is None:
        if diag:
            logger.warning(
                "pipeline.run aborted (%s): %s",
                diag.reason_code,
                diag.message,
            )
        empty = pd.DataFrame()
        if diag:
            empty.attrs["diagnostic"] = diag
        return empty

    res = diag_res.value
    stats = res["out_sample_stats"]
    df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in res.get("benchmark_ir", {}).items():
        col = f"ir_{label}"
        df[col] = pd.Series(
            {k: v for k, v in ir_map.items() if k not in {"equal_weight", "user_weight"}}
        )
    if diag:
        df.attrs["diagnostic"] = diag
    return df


def run_full_from_config(cfg: Any, *, bindings: ConfigBindings) -> PipelineResult:
    """Run the analysis pipeline and return diagnostics plus payload.

    Args:
        cfg: Config instance or mapping compatible with `Config`.
        bindings: Helper bindings that load data, resolve config sections, and
            invoke the analysis core.

    Returns:
        PipelineResult containing the payload, diagnostic info, and optional
        metadata (if the underlying analysis provides it).
    """
    cfg = bindings.unwrap_cfg(cfg)
    preprocessing_section = bindings.cfg_section(cfg, "preprocessing")
    data_settings = bindings.cfg_section(cfg, "data")
    csv_path = bindings.section_get(data_settings, "csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    missing_policy_cfg = bindings.section_get(data_settings, "missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = bindings.section_get(data_settings, "nan_policy")
    missing_limit_cfg = bindings.section_get(data_settings, "missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = bindings.section_get(data_settings, "nan_limit")
    date_column_cfg = bindings.section_get(data_settings, "date_column")
    if date_column_cfg is None:
        date_column_cfg = "Date"

    load_kwargs: dict[str, Any] = {
        "errors": "raise",
        "missing_policy": missing_policy_cfg,
        "missing_limit": missing_limit_cfg,
    }
    if _accepts_keyword(bindings.load_csv, "date_column"):
        load_kwargs["date_column"] = date_column_cfg
    df = bindings.load_csv(csv_path, **load_kwargs)
    df = cast(pd.DataFrame, df)

    bindings.attach_calendar_settings(df, cfg)

    split_cfg = bindings.cfg_section(cfg, "sample_split")
    resolved_split = bindings.resolve_sample_split(df, split_cfg)
    metrics_section = bindings.cfg_section(cfg, "metrics")
    metrics_list = bindings.section_get(metrics_section, "registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import canonical_metric_list

        stats_cfg = bindings.RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    missing_section = bindings.section_get(preprocessing_section, "missing_data")
    if not isinstance(missing_section, Mapping):
        missing_section = None
    policy_spec, limit_spec = bindings.policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
    )

    vol_adjust = bindings.cfg_section(cfg, "vol_adjust")
    run_settings = bindings.cfg_section(cfg, "run")
    portfolio_cfg = bindings.cfg_section(cfg, "portfolio")
    weighting_scheme = _resolve_single_period_weighting_scheme(portfolio_cfg, bindings.section_get)
    robustness_cfg = bindings.section_get(portfolio_cfg, "robustness")
    if not isinstance(robustness_cfg, Mapping):
        robustness_cfg = bindings.cfg_section(cfg, "robustness")
    weight_engine_params = bindings.weight_engine_params_from_robustness(
        weighting_scheme, robustness_cfg
    )
    risk_free_column = bindings.section_get(data_settings, "risk_free_column")
    trend_spec = bindings.build_trend_spec(cfg, vol_adjust)
    lambda_tc_val = bindings.section_get(portfolio_cfg, "lambda_tc", 0.0)
    risk_free_column = bindings.section_get(data_settings, "risk_free_column")
    allow_risk_free_fallback = bindings.section_get(data_settings, "allow_risk_free_fallback")

    diag_res = bindings.invoke_analysis_with_diag(
        df,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        bindings.resolve_target_vol(vol_adjust),
        _resolve_single_period_monthly_cost(portfolio_cfg, run_settings),
        floor_vol=bindings.section_get(vol_adjust, "floor_vol"),
        warmup_periods=int(bindings.section_get(vol_adjust, "warmup_periods", 0) or 0),
        selection_mode=bindings.section_get(portfolio_cfg, "selection_mode", "all"),
        random_n=bindings.section_get(portfolio_cfg, "random_n", 8),
        custom_weights=bindings.section_get(portfolio_cfg, "custom_weights"),
        rank_kwargs=bindings.section_get(portfolio_cfg, "rank"),
        manual_funds=bindings.section_get(portfolio_cfg, "manual_list"),
        indices_list=bindings.section_get(portfolio_cfg, "indices_list"),
        benchmarks=bindings.cfg_value(cfg, "benchmarks"),
        seed=bindings.cfg_value(cfg, "seed", 42),
        weighting_scheme=weighting_scheme,
        constraints=bindings.section_get(portfolio_cfg, "constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=bindings.section_get(vol_adjust, "window"),
        previous_weights=bindings.section_get(portfolio_cfg, "previous_weights"),
        lambda_tc=lambda_tc_val,
        max_turnover=bindings.section_get(portfolio_cfg, "max_turnover"),
        signal_spec=trend_spec,
        regime_cfg=bindings.cfg_section(cfg, "regime"),
        weight_policy=bindings.section_get(portfolio_cfg, "weight_policy"),
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        weight_engine_params=weight_engine_params,
    )
    if isinstance(diag_res, PipelineResult):
        normalized = diag_res
    elif isinstance(diag_res, DiagnosticResult):
        normalized = PipelineResult(
            value=diag_res.value,
            diagnostic=diag_res.diagnostic,
        )
    else:
        payload, diagnostic = coerce_pipeline_result(diag_res)
        metadata = getattr(diag_res, "metadata", None)
        if metadata is not None and not isinstance(metadata, Mapping):
            metadata = None
        normalized = PipelineResult(
            value=payload,
            diagnostic=diagnostic,
            metadata=metadata,
        )

    diag = normalized.diagnostic
    if normalized.value is None:
        if diag:
            logger.warning(
                "pipeline.run_full aborted (%s): %s",
                diag.reason_code,
                diag.message,
            )
        else:
            logger.warning("pipeline.run_full aborted with no diagnostic context")
    return normalized
