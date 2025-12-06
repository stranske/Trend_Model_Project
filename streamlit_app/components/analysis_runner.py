"""Helpers to execute the Trend analysis pipeline from the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from trend_analysis.config.legacy import Config
from trend_analysis.signals import TrendSpec as TrendSpecModel

from .data_cache import cache_key_for_frame

METRIC_REGISTRY = {
    "sharpe": "Sharpe",
    "return_ann": "AnnualReturn",
    "sortino": "Sortino",
    "info_ratio": "InformationRatio",
    "drawdown": "MaxDrawdown",
    "vol": "Volatility",
}


@dataclass(frozen=True)
class ModelSettings:
    """Compatibility shim for legacy demo code expecting ``ModelSettings``."""

    lookback_months: int
    rebalance_frequency: str
    selection_count: int
    risk_target: float
    weighting_scheme: str
    cooldown_months: int
    min_track_months: int
    metric_weights: Mapping[str, float]
    trend_spec: Mapping[str, Any]
    benchmark: str | None = None


@dataclass
class AnalysisPayload:
    """Container describing the data required to run the analysis."""

    returns: pd.DataFrame
    model_state: Mapping[str, Any]
    benchmark: str | None


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(as_int, minimum)


def _coerce_positive_float(value: Any, *, default: float) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return default
    return max(as_float, 0.0)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    period = stamp.to_period("M")
    return period.to_timestamp("M", how="end")


def _build_sample_split(
    index: pd.DatetimeIndex, config: Mapping[str, Any]
) -> dict[str, str]:
    if index.empty:
        raise ValueError("Dataset is empty")

    # Check if user specified explicit date mode
    date_mode = config.get("date_mode", "relative")

    if date_mode == "explicit":
        # User has specified explicit start/end dates
        user_start = config.get("start_date")
        user_end = config.get("end_date")

        if not user_start or not user_end:
            raise ValueError(
                "Explicit date mode requires both start_date and end_date to be specified"
            )

        # Parse user dates
        try:
            start_ts = pd.Timestamp(user_start)
            end_ts = pd.Timestamp(user_end)
        except (ValueError, TypeError):
            # Fall back to relative mode on parse error
            pass
        else:
            # Clamp to data boundaries
            data_start = index.min()
            data_end = index.max()
            start_ts = max(start_ts, data_start)
            end_ts = min(end_ts, data_end)

            # For explicit mode, use lookback_months to determine in-sample split
            lookback_months = _coerce_positive_int(
                config.get("lookback_months"), default=36, minimum=1
            )

            # Calculate date boundaries for explicit mode
            # out_start and out_end come from user-specified dates
            out_start = _month_end(start_ts)
            out_end = _month_end(end_ts)

            # in_end is one month before out_start
            in_end = _month_end(out_start - pd.DateOffset(months=1))
            if in_end < index.min():
                in_end = _month_end(index.min())

            # in_start is lookback_months before in_end
            in_start = _month_end(in_end - pd.DateOffset(months=lookback_months - 1))
            if in_start < index.min():
                in_start = _month_end(index.min())

            return {
                "in_start": in_start.strftime("%Y-%m"),
                "in_end": in_end.strftime("%Y-%m"),
                "out_start": out_start.strftime("%Y-%m"),
                "out_end": out_end.strftime("%Y-%m"),
            }

    # Relative mode (default): compute from lookback/evaluation windows
    lookback_months = _coerce_positive_int(
        config.get("lookback_months"), default=36, minimum=1
    )
    evaluation_months = _coerce_positive_int(
        config.get("evaluation_months"), default=12, minimum=1
    )

    last = _month_end(index.max())
    first = _month_end(index.min())
    out_start = _month_end(last - pd.DateOffset(months=evaluation_months - 1))
    if out_start < first:
        out_start = first
    in_end = _month_end(out_start - pd.DateOffset(months=1))
    if in_end < first:
        in_end = first
    in_start = _month_end(in_end - pd.DateOffset(months=lookback_months - 1))
    if in_start < first:
        in_start = first

    return {
        "in_start": in_start.strftime("%Y-%m"),
        "in_end": in_end.strftime("%Y-%m"),
        "out_start": out_start.strftime("%Y-%m"),
        "out_end": last.strftime("%Y-%m"),
    }


def _build_signals_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = TrendSpecModel()
    window = _coerce_positive_int(config.get("window"), default=base.window)
    lag = _coerce_positive_int(config.get("lag"), default=base.lag)
    min_periods_raw = config.get("min_periods")
    try:
        min_periods = (
            int(min_periods_raw) if min_periods_raw not in (None, "") else None
        )
    except (TypeError, ValueError):
        min_periods = None
    if min_periods is not None and min_periods <= 0:
        min_periods = None
    if min_periods is not None and min_periods > window:
        min_periods = window

    vol_adjust = bool(config.get("vol_adjust", base.vol_adjust))
    vol_target_raw = config.get("vol_target")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
    except (TypeError, ValueError):
        vol_target = None
    if vol_target is not None and vol_target <= 0:
        vol_target = None
    if not vol_adjust:
        vol_target = None

    zscore = bool(config.get("zscore", base.zscore))

    payload: dict[str, Any] = {
        "kind": base.kind,
        "window": window,
        "lag": lag,
        "vol_adjust": vol_adjust,
        "zscore": zscore,
    }
    if min_periods is not None:
        payload["min_periods"] = min_periods
    if vol_target is not None:
        payload["vol_target"] = vol_target
    return payload


def _normalise_metric_weights(raw: Mapping[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name, value in raw.items():
        if name not in METRIC_REGISTRY:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weights[name] = weight
    if not weights:
        default = 1.0 / 3
        return {
            "sharpe": default,
            "return_ann": default,
            "drawdown": default,
        }
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _build_portfolio_config(
    config: Mapping[str, Any], weights: Mapping[str, float]
) -> dict[str, Any]:
    selection_count = _coerce_positive_int(
        config.get("selection_count"), default=10, minimum=1
    )
    weighting_scheme = str(config.get("weighting_scheme", "equal") or "equal")
    registry_weights = {
        METRIC_REGISTRY.get(metric, metric): float(weight)
        for metric, weight in weights.items()
    }

    # Advanced settings
    max_weight = _coerce_positive_float(config.get("max_weight"), default=0.20)
    max_turnover = _coerce_positive_float(config.get("max_turnover"), default=1.0)
    transaction_cost_bps = _coerce_positive_int(
        config.get("transaction_cost_bps"), default=0, minimum=0
    )
    rebalance_freq = str(config.get("rebalance_freq", "M") or "M")

    # Fund holding rules (Phase 3)
    min_tenure_periods = _coerce_positive_int(
        config.get("min_tenure_periods"), default=0, minimum=0
    )
    max_changes_per_period = _coerce_positive_int(
        config.get("max_changes_per_period"), default=0, minimum=0
    )
    max_active_positions = _coerce_positive_int(
        config.get("max_active_positions"), default=0, minimum=0
    )

    portfolio_cfg: dict[str, Any] = {
        "selection_mode": "rank",
        "rank": {
            "inclusion_approach": "top_n",
            "n": selection_count,
            "score_by": "blended",
            "blended_weights": registry_weights,
        },
        "weighting_scheme": weighting_scheme,
        "rebalance_freq": rebalance_freq,
        "max_turnover": max_turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "constraints": {
            "long_only": True,
            "max_weight": max_weight,
        },
    }

    # Add fund holding rules if set (0 means unlimited/disabled)
    if min_tenure_periods > 0:
        portfolio_cfg["min_tenure_n"] = min_tenure_periods
    if max_changes_per_period > 0:
        portfolio_cfg["turnover_budget_max_changes"] = max_changes_per_period
    if max_active_positions > 0:
        portfolio_cfg["max_active"] = max_active_positions

    return portfolio_cfg


def _build_config(payload: AnalysisPayload) -> Config:
    state = payload.model_state
    weights = _normalise_metric_weights(state.get("metric_weights", {}))
    sample_split = _build_sample_split(payload.returns.index, state)
    vol_target = _coerce_positive_float(state.get("risk_target"), default=0.1)

    # Risk settings
    vol_floor = _coerce_positive_float(state.get("vol_floor"), default=0.015)
    warmup_periods = _coerce_positive_int(
        state.get("warmup_periods"), default=0, minimum=0
    )
    rf_rate_annual = _coerce_positive_float(state.get("rf_rate_annual"), default=0.0)

    # Build signals config - use Phase 4 parameters or defaults
    trend_spec = {
        "window": state.get("trend_window"),
        "lag": state.get("trend_lag"),
        "min_periods": state.get("trend_min_periods"),
        "zscore": state.get("trend_zscore"),
        "vol_adjust": state.get("trend_vol_adjust"),
        "vol_target": state.get("trend_vol_target"),
    }
    signals_cfg = _build_signals_config(trend_spec)

    portfolio_cfg = _build_portfolio_config(state, weights)

    metrics_registry = [METRIC_REGISTRY.get(name, name) for name in weights]

    benchmark_map: dict[str, str] = {}
    if payload.benchmark:
        benchmark_map[payload.benchmark] = payload.benchmark

    random_seed_raw = state.get("random_seed")
    seed = 42
    try:
        if random_seed_raw is not None:
            seed = int(random_seed_raw)
    except (TypeError, ValueError):
        seed = 42

    # Get preset name from either new or old format
    preset_name = state.get("preset") or state.get("trend_spec_preset")

    # Regime analysis settings (Phase 6)
    regime_enabled = bool(state.get("regime_enabled", False))
    regime_proxy = str(state.get("regime_proxy", "SPX") or "SPX")
    regime_cfg = {
        "enabled": regime_enabled,
        "proxy": regime_proxy,
    }

    # Robustness settings (Phase 7)
    shrinkage_enabled = bool(state.get("shrinkage_enabled", True))
    shrinkage_method = str(
        state.get("shrinkage_method", "ledoit_wolf") or "ledoit_wolf"
    )
    leverage_cap = _coerce_positive_float(state.get("leverage_cap"), default=2.0)

    robustness_cfg = {
        "shrinkage": {
            "enabled": shrinkage_enabled,
            "method": shrinkage_method,
        },
    }

    # Entry/Exit thresholds (Phase 5)
    z_entry_soft = float(state.get("z_entry_soft", 1.0) or 1.0)
    z_exit_soft = float(state.get("z_exit_soft", -1.0) or -1.0)
    soft_strikes = int(state.get("soft_strikes", 2) or 2)
    entry_soft_strikes = int(state.get("entry_soft_strikes", 1) or 1)
    sticky_add_periods = int(state.get("sticky_add_periods", 1) or 1)
    sticky_drop_periods = int(state.get("sticky_drop_periods", 1) or 1)
    ci_level = float(state.get("ci_level", 0.0) or 0.0)

    # Build threshold_hold config for portfolio
    threshold_hold_cfg = {
        "z_entry_soft": z_entry_soft,
        "z_exit_soft": z_exit_soft,
        "soft_strikes": soft_strikes,
        "entry_soft_strikes": entry_soft_strikes,
    }

    # Add threshold_hold and policy settings to portfolio config
    portfolio_cfg["policy"] = "threshold_hold"
    portfolio_cfg["threshold_hold"] = threshold_hold_cfg

    # Add sticky periods and CI to policy config (for PolicyConfig in simulator)
    portfolio_cfg["sticky_add_x"] = sticky_add_periods
    portfolio_cfg["sticky_drop_y"] = sticky_drop_periods
    portfolio_cfg["ci_level"] = ci_level

    # Update portfolio config with leverage cap
    portfolio_cfg["leverage_cap"] = leverage_cap

    return Config(
        version="1",
        data={"allow_risk_free_fallback": True},
        preprocessing={},
        vol_adjust={
            "target_vol": vol_target,
            "floor_vol": vol_floor,
            "warmup_periods": warmup_periods,
        },
        sample_split=sample_split,
        portfolio=portfolio_cfg,
        signals=signals_cfg,
        benchmarks=benchmark_map,
        regime=regime_cfg,
        robustness=robustness_cfg,
        metrics={
            "registry": metrics_registry,
            "rf_rate_annual": rf_rate_annual,
        },
        export={},
        run={"trend_preset": preset_name},
        seed=seed,
    )


def _prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    reset = df.reset_index()
    index_name = df.index.name or "Date"
    return reset.rename(columns={index_name: "Date"})


def _execute_analysis(payload: AnalysisPayload):
    from trend_analysis.api import run_simulation

    config = _build_config(payload)
    returns = _prepare_returns(payload.returns)
    return run_simulation(config, returns)


def _hashable_model_state(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, default=str)


@st.cache_data(
    show_spinner="Running analysis…", hash_funcs={pd.DataFrame: cache_key_for_frame}
)
def run_cached_analysis(
    returns: pd.DataFrame,
    model_state_blob: str,
    benchmark: str | None,
    data_hash: str,
):
    """
    Run the analysis pipeline with caching.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame containing asset returns, indexed by date.
    model_state_blob : str
        JSON-serialized model state containing analysis configuration.
    benchmark : str or None
        Optional benchmark identifier for the analysis.

    Returns
    -------
    Any
        The result of the analysis pipeline, as returned by `run_simulation`.
    """
    model_state = json.loads(model_state_blob)
    payload = AnalysisPayload(
        returns=returns,
        model_state=model_state,
        benchmark=benchmark,
    )
    return _execute_analysis(payload)


def run_analysis(
    df: pd.DataFrame,
    model_state: Mapping[str, Any],
    benchmark: str | None,
    *,
    data_hash: str | None = None,
):
    """Execute the cached analysis pipeline."""

    blob = _hashable_model_state(model_state)
    effective_hash = data_hash or cache_key_for_frame(df)
    return run_cached_analysis(df, blob, benchmark, effective_hash)


def clear_cached_analysis() -> None:
    """Invalidate any cached analysis results."""

    clear_fn = getattr(run_cached_analysis, "clear", None)
    if callable(clear_fn):
        clear_fn()
