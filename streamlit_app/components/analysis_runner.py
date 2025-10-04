"""Helpers for preparing inputs and running cached preprocessing for the app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from trend_analysis.config import Config
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig

PIPELINE_METRIC_ALIASES: Mapping[str, str] = {
    "sharpe": "Sharpe",
    "return_ann": "AnnualReturn",
    "drawdown": "MaxDrawdown",
    "vol": "Volatility",
}


@dataclass(frozen=True)
class ModelSettings:
    """Persistent model settings captured from the Model page."""

    lookback_months: int
    rebalance_frequency: str
    selection_count: int
    risk_target: float
    weighting_scheme: str
    cooldown_months: int
    min_track_months: int
    metric_weights: Mapping[str, float]
    trend_spec: Mapping[str, Any]
    benchmark: str | None


@st.cache_data(show_spinner=False)
def prepare_returns_panel(
    df: pd.DataFrame,
    *,
    date_index_name: str | None,
    return_columns: Sequence[str],
    benchmark_column: str | None,
) -> pd.DataFrame:
    """Return a returns panel ready for the simulation pipeline."""

    work = df.copy()
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()

    keep_cols = list(return_columns)
    if benchmark_column and benchmark_column not in keep_cols:
        keep_cols.append(benchmark_column)
    panel = work[keep_cols].copy()
    panel.reset_index(inplace=True)
    panel.rename(columns={panel.columns[0]: date_index_name or "Date"}, inplace=True)
    panel[panel.columns[0]] = pd.to_datetime(panel[panel.columns[0]]).dt.normalize()
    return panel


def _normalise_metric_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    normed: Dict[str, float] = {}
    for metric, weight in weights.items():
        try:
            val = float(weight)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        key = str(metric).lower()
        normed[key] = val
    total = float(sum(normed.values()))
    if total <= 0:
        default = 1.0 / 3
        return {"sharpe": default, "return_ann": default, "drawdown": default}
    return {name: val / total for name, val in normed.items()}


def build_policy_config(settings: ModelSettings) -> PolicyConfig:
    weights = _normalise_metric_weights(settings.metric_weights)
    metrics = [
        MetricSpec(name=metric, weight=weight)
        for metric, weight in weights.items()
    ]
    return PolicyConfig(
        top_k=settings.selection_count,
        bottom_k=0,
        cooldown_months=settings.cooldown_months,
        min_track_months=settings.min_track_months,
        max_active=max(settings.selection_count * 2, 50),
        max_weight=0.15,
        metrics=metrics,
    )


@st.cache_data(show_spinner=False)
def derive_analysis_window(
    index: Iterable[pd.Timestamp], *, lookback_months: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Determine the analysis window given an index and lookback."""

    index = pd.to_datetime(pd.Index(list(index))).sort_values()
    if index.empty:
        raise ValueError("Dataset has no rows")
    periods = index.to_period("M")
    unique = periods.unique().sort_values()
    if len(unique) <= lookback_months + 2:
        raise ValueError(
            "Not enough history for the selected lookback. Reduce the window length."
        )
    # Find the first date in the index matching the start period
    start_period = unique[lookback_months]
    end_period = unique[-1]
    start = index[periods == start_period][0]
    end = index[periods == end_period][-1]
    return start, end


def build_pipeline_config(
    *,
    settings: ModelSettings,
    policy: PolicyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark: str | None,
) -> Config:
    """Construct a Trend pipeline ``Config`` for the run."""

    lookback = int(settings.lookback_months)
    sample_split = {
        "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
        "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
        "out_start": start.strftime("%Y-%m"),
        "out_end": end.strftime("%Y-%m"),
    }

    weights = _normalise_metric_weights(settings.metric_weights)
    blended_weights = {
        PIPELINE_METRIC_ALIASES.get(metric, metric): float(weight)
        for metric, weight in weights.items()
    }

    registry = list(blended_weights.keys())
    portfolio = {
        "selection_mode": "rank",
        "rank": {
            "inclusion_approach": "top_n",
            "n": settings.selection_count,
            "score_by": "blended",
            "blended_weights": blended_weights,
        },
        "weighting_scheme": settings.weighting_scheme,
    }
    benchmarks = {benchmark: benchmark} if benchmark else {}

    config = Config(
        version="1",
        data={},
        preprocessing={"trend": dict(settings.trend_spec)},
        vol_adjust={
            "target_vol": float(settings.risk_target),
            "floor_vol": 0.015,
            "warmup_periods": 0,
        },
        sample_split=sample_split,
        portfolio=portfolio,
        benchmarks=benchmarks,
        metrics={"registry": registry},
        export={},
        run={"trend_preset": None},
    )
    config.policy = policy.dict()
    return config


def clear_preprocessing_caches() -> None:
    """Clear cached preprocessing helpers."""

    prepare_returns_panel.clear()
    derive_analysis_window.clear()
