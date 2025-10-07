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
    return {
        "selection_mode": "rank",
        "rank": {
            "inclusion_approach": "top_n",
            "n": selection_count,
            "score_by": "blended",
            "blended_weights": registry_weights,
        },
        "weighting_scheme": weighting_scheme,
    }


def _build_config(payload: AnalysisPayload) -> Config:
    state = payload.model_state
    weights = _normalise_metric_weights(state.get("metric_weights", {}))
    sample_split = _build_sample_split(payload.returns.index, state)
    vol_target = _coerce_positive_float(state.get("risk_target"), default=0.1)
    signals_cfg = _build_signals_config(state.get("trend_spec", {}))
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

    return Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={
            "target_vol": vol_target,
            "floor_vol": 0.015,
            "warmup_periods": int(state.get("warmup_periods", 0) or 0),
        },
        sample_split=sample_split,
        portfolio=portfolio_cfg,
        signals=signals_cfg,
        benchmarks=benchmark_map,
        metrics={"registry": metrics_registry},
        export={},
        run={"trend_preset": state.get("trend_spec_preset")},
        seed=seed,
    )


def _prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    reset = df.reset_index()
    index_name = df.index.name or "Date"
    return reset.rename(columns={index_name: "Date"})


def _run_analysis(payload: AnalysisPayload):
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
    returns: pd.DataFrame, model_state_blob: str, benchmark: str | None
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
    return _run_analysis(payload)


def run_analysis(
    df: pd.DataFrame, model_state: Mapping[str, Any], benchmark: str | None
):
    """Execute the cached analysis pipeline."""

    blob = _hashable_model_state(model_state)
    return run_cached_analysis(df, blob, benchmark)


def clear_cached_analysis() -> None:
    """Invalidate any cached analysis results."""

    clear_fn = getattr(run_cached_analysis, "clear", None)
    if callable(clear_fn):
        clear_fn()
