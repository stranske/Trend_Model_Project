from __future__ import annotations

import logging
import random
import sys
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as ConfigType
else:  # Runtime: avoid importing typing-only names
    from typing import Any as ConfigType

from .logging import log_step as _log_step  # lightweight import
from .pipeline import _policy_from_config, _resolve_sample_split, _run_analysis
from trend.validation import (
    assert_execution_lag,
    build_validation_frame,
    validate_prices_frame,
)

logger = logging.getLogger(__name__)


def _safe_len(obj: Any) -> int:
    """Return len(obj) when supported, otherwise zero."""

    return len(obj) if isinstance(obj, Sized) else 0


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
    details_sanitized: Any | None = None


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

    validation_frame = validate_prices_frame(build_validation_frame(returns))

    data_settings = getattr(config, "data", {}) or {}
    max_lag_days = data_settings.get("max_lag_days")
    lag_limit: int | None = None
    if max_lag_days not in (None, ""):
        try:
            lag_limit = int(max_lag_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("data.max_lag_days must be an integer") from exc

    split = config.sample_split
    metrics_list = config.metrics.get("registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    regime_cfg = getattr(config, "regime", {}) or {}

    preprocessing_section = getattr(config, "preprocessing", {}) or {}
    missing_section = (
        preprocessing_section.get("missing_data")
        if isinstance(preprocessing_section, Mapping)
        else None
    )
    policy_spec, limit_spec = _policy_from_config(
        missing_section if isinstance(missing_section, Mapping) else None
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

    res = _run_analysis(
        returns,
        resolved_split["in_start"],
        resolved_split["in_end"],
        resolved_split["out_start"],
        resolved_split["out_end"],
        config.vol_adjust.get("target_vol", 1.0),
        getattr(config, "run", {}).get("monthly_cost", 0.0),
        floor_vol=config.vol_adjust.get("floor_vol"),
        warmup_periods=int(config.vol_adjust.get("warmup_periods", 0) or 0),
        selection_mode=config.portfolio.get("selection_mode", "all"),
        random_n=config.portfolio.get("random_n", 8),
        custom_weights=config.portfolio.get("custom_weights"),
        rank_kwargs=config.portfolio.get("rank"),
        manual_funds=config.portfolio.get("manual_list"),
        indices_list=config.portfolio.get("indices_list"),
        benchmarks=config.benchmarks,
        seed=seed,
        weighting_scheme=config.portfolio.get("weighting_scheme", "equal"),
        constraints=config.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
        missing_policy=policy_spec,
        missing_limit=limit_spec,
        risk_window=config.vol_adjust.get("window"),
        previous_weights=config.portfolio.get("previous_weights"),
        max_turnover=config.portfolio.get("max_turnover"),
        regime_cfg=regime_cfg,
    )
    if res is None:
        logger.warning("run_simulation produced no result")
        return RunResult(pd.DataFrame(), {}, seed, env)

    if isinstance(res, dict):
        res_dict: dict[str, Any] = res
    elif isinstance(res, Mapping):
        res_dict = dict(res)
    else:
        logger.warning("Unexpected result type from _run_analysis: %s", type(res))
        return RunResult(pd.DataFrame(), {}, seed, env)

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
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )

    fallback_raw = res_dict.get("weight_engine_fallback")
    fallback_info: dict[str, Any] | None = (
        fallback_raw if isinstance(fallback_raw, dict) else None
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
    _log_step(
        run_id, "api_end", "run_simulation complete", fallback=bool(fallback_info)
    )
    # Construct portfolio series for bundle export (equal-weight baseline)
    try:
        in_scaled = res_dict.get("in_sample_scaled")
        out_scaled = res_dict.get("out_sample_scaled")
        ew_weights = res_dict.get("ew_weights")
        if (
            isinstance(in_scaled, pd.DataFrame)
            and isinstance(out_scaled, pd.DataFrame)
            and isinstance(ew_weights, dict)
        ):
            # Build one continuous portfolio series across IS + OS
            import numpy as _np

            from .pipeline import calc_portfolio_returns as _cpr

            cols = list(in_scaled.columns)
            w = _np.array([ew_weights.get(c, 0.0) for c in cols])
            port_is = _cpr(w, in_scaled)
            port_os = _cpr(w, out_scaled)
            portfolio_series = pd.concat([port_is, port_os])
            res_dict["portfolio_equal_weight_combined"] = portfolio_series
    except (
        KeyError,
        AttributeError,
        TypeError,
        IndexError,
    ):  # pragma: no cover - defensive
        pass

    rr = RunResult(
        metrics=metrics_df,
        details=res_dict,
        seed=seed,
        environment=env,
        fallback_info=fallback_info,
    )
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
