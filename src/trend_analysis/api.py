from __future__ import annotations

import logging
import random
import sys
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, SupportsInt, cast

import numpy as np
import pandas as pd

from analysis import Results
from trend.diagnostics import DiagnosticPayload

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as ConfigType
else:  # Runtime: avoid importing typing-only names
    from typing import Any as ConfigType

from trend.validation import (
    assert_execution_lag,
    build_validation_frame,
    validate_prices_frame,
)

from .diagnostics import PipelineReasonCode, coerce_pipeline_result
from .logging import log_step as _log_step  # lightweight import
from .pipeline import (
    _build_trend_spec,
    _policy_from_config,
    _resolve_sample_split,
    _resolve_target_vol,
    _run_analysis_with_diagnostics,
)
from .util.risk_free import resolve_risk_free_settings
from .weights.robust_config import weight_engine_params_from_robustness


def _run_analysis(
    *args: Any,
    signals_cfg: Mapping[str, Any] | None = None,
    vol_adjust_cfg: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Back-compat hook for tests while wiring signals into the pipeline."""

    if signals_cfg is not None and "signal_spec" not in kwargs:
        trend_spec = _build_trend_spec({"signals": signals_cfg}, vol_adjust_cfg or {})
        kwargs["signal_spec"] = trend_spec
    return _run_analysis_with_diagnostics(*args, **kwargs)


logger = logging.getLogger(__name__)


def _safe_len(obj: Any) -> int:
    """Return len(obj) when supported, otherwise zero."""

    return len(obj) if isinstance(obj, Sized) else 0


def _attach_reporting_metadata(res_dict: dict[str, Any], config: ConfigType) -> None:
    """Attach reporting-only metadata from config without affecting computation."""

    portfolio = getattr(config, "portfolio", None)
    if not isinstance(portfolio, Mapping):
        return
    ci_level = portfolio.get("ci_level")
    if ci_level is None:
        return
    if isinstance(ci_level, str) and ci_level == "":
        return
    try:
        ci_level_val = float(ci_level)
    except (TypeError, ValueError):
        return
    if ci_level_val <= 0:
        return
    metadata = res_dict.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        res_dict["metadata"] = metadata
    reporting = metadata.get("reporting")
    if not isinstance(reporting, dict):
        reporting = {}
        metadata["reporting"] = reporting
    reporting["ci_level"] = ci_level_val


@dataclass
class RunResult:
    """Container for simulation output.

    Attributes
    ----------
    metrics : pd.DataFrame
        Summary metrics table.
    details : dict[str, Any]
        Full result payload returned by the pipeline.
    seed : int
        Random seed used for reproducibility.
    environment : dict[str, Any]
        Environment metadata (python/numpy/pandas versions).
    fallback_info : dict[str, Any] | None
        Present when a requested weight engine failed and the system
        reverted to equal weights.  Includes keys: ``engine``,
        ``error_type`` and ``error``.
    """

    metrics: pd.DataFrame
    details: dict[str, Any]
    seed: int
    environment: dict[str, Any]
    fallback_info: dict[str, Any] | None = None
    analysis: Results | None = None
    portfolio: pd.Series | None = None
    weights: pd.Series | None = None
    exposures: pd.Series | None = None
    turnover: pd.Series | None = None
    costs: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None
    details_sanitized: Any | None = None
    diagnostic: DiagnosticPayload | None = None
    # Multi-period specific fields
    period_results: list[dict[str, Any]] | None = None
    period_count: int = 0


def _run_multi_period_simulation(
    config: ConfigType,
    returns: pd.DataFrame,
    env: dict[str, Any],
    seed: int,
) -> RunResult:
    """Execute multi-period simulation and aggregate results.

    Parameters
    ----------
    config : Config
        Configuration object with multi_period settings.
    returns : pd.DataFrame
        DataFrame of returns including a ``Date`` column.
    env : dict
        Environment metadata.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    RunResult
        Aggregated results from all periods.
    """
    from .export import combined_summary_result
    from .multi_period import run as run_multi_period

    run_id = getattr(config, "run_id", None) or "api_multi_run"
    _log_step(run_id, "multi_period_start", "Starting multi-period simulation")

    try:
        period_results = run_multi_period(config, returns)
    except Exception as exc:
        logger.error("Multi-period simulation failed: %s", exc)
        return RunResult(
            metrics=pd.DataFrame(),
            details={"error": str(exc)},
            seed=seed,
            environment=env,
        )

    if not period_results:
        logger.warning("Multi-period simulation returned no results")
        return RunResult(
            metrics=pd.DataFrame(),
            details={},
            seed=seed,
            environment=env,
        )

    _log_step(
        run_id,
        "multi_period_complete",
        f"Multi-period simulation complete with {len(period_results)} periods",
    )

    # Build combined turnover series from all periods
    turnover_series = _build_multi_period_turnover(period_results)

    # Build combined portfolio returns series
    portfolio_series = _build_multi_period_portfolio(period_results)

    # Aggregate results across all periods (may fail if period results lack keys)
    try:
        summary = combined_summary_result(period_results)
        summary = dict(summary)
    except Exception as exc:
        logger.warning("Failed to aggregate multi-period results: %s", exc)
        summary = {}

    # Build metrics DataFrame from aggregated stats
    stats_obj = summary.get("out_sample_stats")
    if isinstance(stats_obj, dict):
        stats_items = list(stats_obj.items())
    else:
        stats_items = list(getattr(stats_obj, "items", lambda: [])())

    metrics_df = pd.DataFrame()
    if stats_items:
        try:
            metrics_df = pd.DataFrame({k: vars(v) for k, v in stats_items}).T
        except Exception as exc:
            logger.warning("Failed to build metrics DataFrame: %s", exc)

    # Combine all period details into the summary
    details = dict(summary)
    details["period_results"] = period_results
    details["period_count"] = len(period_results)
    if portfolio_series is not None:
        details["portfolio_equal_weight_combined"] = portfolio_series
    if turnover_series is not None:
        details["turnover"] = turnover_series

    # Build structured Results object if possible
    structured: Results | None = None
    try:
        structured = Results.from_payload(details)
    except Exception as exc:
        logger.debug("Failed to build structured Results for multi-period: %s", exc)

    rr = RunResult(
        metrics=metrics_df,
        details=details,
        seed=seed,
        environment=env,
        analysis=structured,
        turnover=turnover_series,
        portfolio=portfolio_series,
        period_results=period_results,
        period_count=len(period_results),
    )

    if structured is not None:
        try:
            rr.weights = structured.weights
            rr.exposures = structured.exposures
            if rr.turnover is None:
                rr.turnover = structured.turnover
            rr.costs = dict(structured.costs)
            rr.metadata = structured.metadata
        except Exception:
            pass

    return rr


def _build_multi_period_turnover(
    period_results: list[dict[str, Any]],
) -> pd.Series | None:
    """Build a combined turnover series from multi-period results."""
    turnover_data: dict[str, float] = {}

    for res in period_results:
        period = res.get("period")
        if period is None:
            continue
        # Use out-sample start date as the rebalance date
        out_start = period[2] if len(period) > 2 else None
        if out_start is None:
            continue

        # Try to extract turnover from various possible locations
        turnover_val = res.get("turnover")
        if turnover_val is None:
            risk_diag = res.get("risk_diagnostics")
            if isinstance(risk_diag, dict):
                turnover_val = risk_diag.get("turnover")

        if isinstance(turnover_val, (int, float)):
            turnover_data[out_start] = float(turnover_val)
        elif isinstance(turnover_val, pd.Series) and not turnover_val.empty:
            # Take the last turnover value from this period
            turnover_data[out_start] = float(turnover_val.iloc[-1])

    if not turnover_data:
        return None

    return pd.Series(turnover_data, name="turnover").sort_index()


def _build_multi_period_portfolio(
    period_results: list[dict[str, Any]],
) -> pd.Series | None:
    """Build combined portfolio returns from multi-period out-sample results.

    Uses the actual fund weights applied during the simulation (not equal weights)
    to compute the weighted portfolio returns for each out-of-sample period.
    """
    from .pipeline import calc_portfolio_returns

    out_series_list: list[pd.Series] = []

    for res in period_results:
        user_series = res.get("portfolio_user_weight")
        if isinstance(user_series, pd.Series) and not user_series.empty:
            out_series_list.append(user_series.astype(float))
            continue

        out_df = res.get("out_sample_scaled")
        # Use actual fund weights (user weights) instead of equal weights
        # fund_weights contains the weights actually applied during the simulation
        fund_weights = res.get("fund_weights", {})
        # Fall back to ew_weights only if fund_weights is empty
        if not fund_weights:
            fund_weights = res.get("ew_weights", {})

        if not isinstance(out_df, pd.DataFrame) or out_df.empty:
            continue
        if not fund_weights:
            continue

        try:
            cols = list(out_df.columns)
            w = np.array([fund_weights.get(c, 0.0) for c in cols])
            port_ret = calc_portfolio_returns(w, out_df)
            out_series_list.append(port_ret)
        except Exception:
            continue

    if not out_series_list:
        return None

    return pd.concat(out_series_list).sort_index()


def _build_combined_portfolio_series(
    weights: Mapping[str, float] | None,
    in_df: pd.DataFrame | None,
    out_df: pd.DataFrame | None,
) -> pd.Series | None:
    """Build a combined in/out-sample portfolio series from weights."""
    if not isinstance(weights, Mapping) or not weights:
        return None
    if not isinstance(in_df, pd.DataFrame) or not isinstance(out_df, pd.DataFrame):
        return None
    if in_df.empty or out_df.empty:
        return None

    from .pipeline import calc_portfolio_returns

    try:
        in_cols = list(in_df.columns)
        in_weights = np.array([weights.get(c, 0.0) for c in in_cols])
        port_is = calc_portfolio_returns(in_weights, in_df)

        out_cols = list(out_df.columns)
        out_weights = np.array([weights.get(c, 0.0) for c in out_cols])
        port_os = calc_portfolio_returns(out_weights, out_df)
    except Exception:
        return None

    combined = pd.concat([port_is, port_os])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def run_simulation(config: ConfigType, returns: pd.DataFrame) -> RunResult:
    """Execute the analysis pipeline using pre-loaded returns data.

    Parameters
    ----------
    config : Config
        Configuration object controlling the run.
    returns : pd.DataFrame
        DataFrame of returns including a ``Date`` column.

    Returns
    -------
    RunResult
        Structured results with the summary metrics and detailed payload.
    """
    logger.info("run_simulation start")
    run_id = getattr(config, "run_id", None) or "api_run"
    _log_step(run_id, "api_start", "run_simulation invoked")

    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    # Check for multi-period mode and delegate if enabled
    multi_period_cfg = getattr(config, "multi_period", None)
    if multi_period_cfg is not None and isinstance(multi_period_cfg, dict):
        return _run_multi_period_simulation(config, returns, env, seed)

    validation_frame = validate_prices_frame(build_validation_frame(returns))

    data_settings = getattr(config, "data", {}) or {}
    risk_free_column, allow_risk_free_fallback = resolve_risk_free_settings(data_settings)
    max_lag_days = data_settings.get("max_lag_days")
    lag_limit: int | None = None
    if max_lag_days not in (None, ""):
        try:
            as_int_like = cast(SupportsInt | str, max_lag_days)
            lag_limit = int(as_int_like)
        except (TypeError, ValueError) as exc:
            raise ValueError("data.max_lag_days must be an integer") from exc

    split = config.sample_split
    metrics_list = config.metrics.get("registry")
    # Use rf_rate_annual from config as fallback when override is enabled
    rf_override_enabled = config.metrics.get("rf_override_enabled", False)
    rf_rate_fallback = (
        float(config.metrics.get("rf_rate_annual", 0.0)) if rf_override_enabled else 0.0
    )
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=rf_rate_fallback,
        )

    regime_cfg = getattr(config, "regime", {}) or {}
    vol_adjust_cfg = getattr(config, "vol_adjust", {}) or {}
    signals_cfg = getattr(config, "signals", None)
    if not isinstance(signals_cfg, Mapping):
        signals_cfg = None

    preprocessing_section = getattr(config, "preprocessing", {}) or {}
    missing_section = (
        preprocessing_section.get("missing_data")
        if isinstance(preprocessing_section, Mapping)
        else None
    )
    policy_spec, limit_spec = _policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
    )

    weighting_scheme = config.portfolio.get("weighting_scheme", "equal")
    robustness_cfg = config.portfolio.get("robustness")
    if not isinstance(robustness_cfg, Mapping):
        robustness_cfg = getattr(config, "robustness", None)
    weight_engine_params = weight_engine_params_from_robustness(
        weighting_scheme,
        robustness_cfg if isinstance(robustness_cfg, Mapping) else None,
    )

    if lag_limit is not None:
        as_of_candidate = (
            data_settings.get("as_of")
            or data_settings.get("as_of_date")
            or split.get("out_end")
            or split.get("in_end")
        )
        assert_execution_lag(
            validation_frame,
            as_of=as_of_candidate,
            max_lag_days=lag_limit,
        )

    _log_step(run_id, "analysis_start", "_run_analysis dispatch")
    resolved_split = _resolve_sample_split(returns, split)

    pipeline_output = _run_analysis(
        returns,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        _resolve_target_vol(vol_adjust_cfg),
        getattr(config, "run", {}).get("monthly_cost", 0.0),
        floor_vol=vol_adjust_cfg.get("floor_vol"),
        warmup_periods=int(vol_adjust_cfg.get("warmup_periods", 0) or 0),
        selection_mode=config.portfolio.get("selection_mode", "all"),
        random_n=config.portfolio.get("random_n", 8),
        custom_weights=config.portfolio.get("custom_weights"),
        rank_kwargs=config.portfolio.get("rank"),
        manual_funds=config.portfolio.get("manual_list"),
        indices_list=config.portfolio.get("indices_list"),
        benchmarks=config.benchmarks,
        seed=seed,
        weighting_scheme=weighting_scheme,
        constraints=config.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=vol_adjust_cfg.get("window"),
        previous_weights=config.portfolio.get("previous_weights"),
        lambda_tc=config.portfolio.get("lambda_tc"),
        max_turnover=config.portfolio.get("max_turnover"),
        signals_cfg=signals_cfg,
        vol_adjust_cfg=vol_adjust_cfg,
        regime_cfg=regime_cfg,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=allow_risk_free_fallback,
        weight_engine_params=weight_engine_params,
    )
    diag_hint = cast(DiagnosticPayload | None, getattr(pipeline_output, "diagnostic", None))
    try:
        payload, diag = coerce_pipeline_result(pipeline_output)
    except TypeError as exc:
        logger.warning(
            "Unexpected pipeline result type (%s); returning empty payload",
            exc,
        )
        return RunResult(pd.DataFrame(), {}, seed, env, diagnostic=diag_hint)
    if payload is None:
        # Prefer NO_FUNDS_SELECTED when the input has no investable fund columns
        # (e.g. Date + RF only), even if the configured split yields an empty
        # window. This is the most actionable diagnostic for API callers.
        if diag and diag.reason_code == PipelineReasonCode.SAMPLE_WINDOW_EMPTY.value:
            date_col = str(data_settings.get("date_column", "Date") or "Date")
            excluded = {date_col}
            if risk_free_column:
                excluded.add(str(risk_free_column))
            indices_list = config.portfolio.get("indices_list")
            if isinstance(indices_list, list):
                excluded |= {str(x) for x in indices_list}
            investable_cols = [c for c in returns.columns if str(c) not in excluded]
            if not investable_cols:
                diag = DiagnosticPayload(
                    reason_code=PipelineReasonCode.NO_FUNDS_SELECTED.value,
                    message="No investable funds satisfy the selection filters.",
                    context=getattr(diag, "context", None),
                )

        if diag:
            logger.warning(
                "run_simulation produced no result (%s): %s",
                diag.reason_code,
                diag.message,
            )
        else:
            logger.warning("run_simulation produced no result (unknown reason)")
        return RunResult(pd.DataFrame(), {}, seed, env, diagnostic=diag)
    res_dict = payload
    if isinstance(res_dict, dict):
        _attach_reporting_metadata(res_dict, config)

    _log_step(run_id, "metrics_build", "Building metrics dataframe")
    stats_obj = res_dict.get("out_sample_stats")
    if isinstance(stats_obj, dict):
        stats_items = list(stats_obj.items())
    else:
        stats_items = list(getattr(stats_obj, "items", lambda: [])())
    metrics_df = pd.DataFrame({k: vars(v) for k, v in stats_items}).T
    bench_ir_obj = res_dict.get("benchmark_ir", {})
    bench_ir_items = bench_ir_obj.items() if isinstance(bench_ir_obj, dict) else []
    for label, ir_map in bench_ir_items:
        col = f"ir_{label}"
        metrics_df[col] = pd.Series(
            {k: v for k, v in ir_map.items() if k not in {"equal_weight", "user_weight"}}
        )

    fallback_raw = res_dict.get("weight_engine_fallback")
    fallback_info: dict[str, Any] | None = fallback_raw if isinstance(fallback_raw, dict) else None
    if fallback_info:
        logger.warning(
            "Weight engine fallback used (engine=%s, safe_mode=%s, "
            "condition_number=%s, threshold=%s).",
            fallback_info.get("engine"),
            fallback_info.get("safe_mode"),
            fallback_info.get("condition_number"),
            fallback_info.get("condition_threshold"),
        )
    # Granular logging (best-effort; keys may vary by configuration)
    try:  # pragma: no cover - observational logging
        if res_dict.get("selected_funds") is not None:
            _log_step(
                run_id,
                "selection",
                "Funds selected",
                count=_safe_len(res_dict.get("selected_funds")),
            )
            if res_dict.get("weights_user_weight") is not None:
                _log_step(
                    run_id,
                    "weighting",
                    "User weighting applied",
                    n=_safe_len(res_dict.get("weights_user_weight")),
                )
            elif res_dict.get("weights_equal_weight") is not None:
                _log_step(
                    run_id,
                    "weighting",
                    "Equal weighting applied",
                    n=_safe_len(res_dict.get("weights_equal_weight")),
                )
            else:
                # Fallback: approximate number of assets from selected funds
                sel = res_dict.get("selected_funds")
                _log_step(
                    run_id,
                    "weighting",
                    "Implicit equal weighting (no explicit weights recorded)",
                    n=_safe_len(sel),
                )
        if res_dict.get("benchmark_ir") is not None:
            _log_step(
                run_id,
                "benchmarks",
                "Benchmark IR computed",
                n=_safe_len(res_dict.get("benchmark_ir")),
            )
    except Exception:
        pass
    logger.info("run_simulation end")
    _log_step(run_id, "api_end", "run_simulation complete", fallback=bool(fallback_info))
    # Construct portfolio series for bundle export (equal-weight baseline)
    try:
        in_scaled = res_dict.get("in_sample_scaled")
        out_scaled = res_dict.get("out_sample_scaled")
        ew_weights = res_dict.get("ew_weights")
        portfolio_series = _build_combined_portfolio_series(
            ew_weights,
            in_scaled,
            out_scaled,
        )
        if portfolio_series is not None:
            res_dict["portfolio_equal_weight_combined"] = portfolio_series

        fund_weights = res_dict.get("fund_weights")
        user_series = _build_combined_portfolio_series(
            fund_weights,
            in_scaled,
            out_scaled,
        )
        if user_series is not None:
            res_dict["portfolio_user_weight_combined"] = user_series
    except (
        KeyError,
        AttributeError,
        TypeError,
        IndexError,
    ):  # pragma: no cover - defensive
        pass

    structured: Results | None = None
    try:
        structured = Results.from_payload(res_dict)
    except Exception as exc:  # pragma: no cover - defensive capture
        logger.debug("Failed to build structured Results payload: %s", exc)

    rr = RunResult(
        metrics=metrics_df,
        details=res_dict,
        seed=seed,
        environment=env,
        fallback_info=fallback_info,
        analysis=structured,
        diagnostic=diag,
    )

    if structured is not None:
        try:
            rr.portfolio = structured.returns
            rr.weights = structured.weights
            rr.exposures = structured.exposures
            rr.turnover = structured.turnover
            rr.costs = dict(structured.costs)
            rr.metadata = structured.metadata
        except Exception:  # pragma: no cover - defensive attribute binding
            pass
    # Ensure details dict is JSON-friendly (no Timestamp / non-primitive keys)
    try:  # pragma: no cover - lightweight sanitation (non-destructive)
        from pandas import DataFrame as _DataFrame
        from pandas import Series as _Series

        def _stringify_key(value: Any) -> str | int | float | bool | None:
            """Return a JSON-friendly representation of ``value``."""

            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, tuple):
                parts = [_stringify_key(part) for part in value]
                return " / ".join(str(part) for part in parts)
            try:
                iso = getattr(value, "isoformat")
            except AttributeError:
                pass
            else:
                try:
                    return str(iso())
                except Exception:  # pragma: no cover - defensive
                    return str(value)
            return str(value)

        def _sanitize_keys(obj: Any) -> Any:
            if isinstance(obj, _Series):
                return {_stringify_key(i): _sanitize_keys(v) for i, v in obj.items()}
            if isinstance(obj, _DataFrame):
                sanitized: dict[str | int | float | bool | None, Any] = {}
                for col in obj.columns:
                    sanitized[_stringify_key(col)] = _sanitize_keys(obj[col])
                return sanitized
            if isinstance(obj, dict):
                new: dict[str | int | float | bool | None, Any] = {}
                for k, v in obj.items():
                    new[_stringify_key(k)] = _sanitize_keys(v)
                return new
            if isinstance(obj, (list, tuple)):
                return [_sanitize_keys(x) for x in obj]
            return obj

        # Store a parallel sanitized view for hashing/export without mutating original
        rr.details_sanitized = _sanitize_keys(rr.details)
    except Exception:  # pragma: no cover
        pass
    return rr
