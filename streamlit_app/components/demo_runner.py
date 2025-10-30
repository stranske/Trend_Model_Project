"""Helpers for running the end-to-end demo pipeline from the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import pandas as pd
import yaml

from streamlit_app.components.analysis_runner import ModelSettings
from trend_analysis.api import run_simulation
from trend_analysis.config import Config
from trend_portfolio_app.data_schema import (
    SchemaMeta,
    infer_benchmarks,
    load_and_validate_file,
)
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "demo"
PRESET_DIR = REPO_ROOT / "config" / "presets"

DEMO_DATA_CANDIDATES = (
    DEMO_DIR / "demo_returns.csv",
    DEMO_DIR / "demo_returns.xlsx",
)

DEFAULT_PRESET = "Balanced"

UI_METRIC_ALIASES: Mapping[str, str] = {
    "sharpe_ratio": "sharpe",
    "sharpe": "sharpe",
    "return_ann": "return_ann",
    "annual_return": "return_ann",
    "max_drawdown": "drawdown",
    "drawdown": "drawdown",
    "volatility": "vol",
    "vol": "vol",
}

PIPELINE_METRIC_ALIASES: Mapping[str, str] = {
    "sharpe": "Sharpe",
    "return_ann": "AnnualReturn",
    "drawdown": "MaxDrawdown",
    "vol": "Volatility",
}


@dataclass
class DemoSetup:
    """Container describing the derived configuration for the demo run."""

    config_state: Dict[str, Any]
    sim_config: Dict[str, Any]
    pipeline_config: Config
    benchmark: str | None


def _load_demo_returns() -> Tuple[pd.DataFrame, SchemaMeta]:
    """Load the built-in demo returns from disk."""

    for path in DEMO_DATA_CANDIDATES:
        if path.exists():
            with path.open("rb") as handle:
                df, meta = load_and_validate_file(handle)
            return df, meta
    raise FileNotFoundError(
        "Demo returns not found. Expected demo/demo_returns.(csv|xlsx)."
    )


def _load_preset(name: str) -> Dict[str, Any]:
    """Load a preset YAML file into a mapping."""

    preset_path = PRESET_DIR / f"{name.lower()}.yml"
    if not preset_path.exists():
        return {}
    with preset_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _select_benchmark(columns: Iterable[str]) -> str | None:
    candidates = infer_benchmarks(list(columns))
    if not candidates:
        return None
    for cand in candidates:
        if cand.upper().startswith("SPX"):
            return cand
    return candidates[0]


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    period = pd.Period(ts, freq="M")
    return period.to_timestamp("M", how="end")


def _derive_window(
    df: pd.DataFrame, lookback_months: int, oos_months: int = 12
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = _month_end(pd.Timestamp(df.index.max()))
    start = _month_end(end - pd.DateOffset(months=max(oos_months - 1, 0)))
    earliest = _month_end(
        pd.Timestamp(df.index.min()) + pd.DateOffset(months=lookback_months)
    )
    if start < earliest:
        start = earliest
    if start > end:
        start = end
    return start, end


def _build_policy(
    metric_weights: Mapping[str, float], preset: Mapping[str, Any]
) -> PolicyConfig:
    metrics = [
        MetricSpec(name=metric, weight=float(weight))
        for metric, weight in metric_weights.items()
    ]
    return PolicyConfig(
        top_k=int(preset.get("selection_count", 10)),
        bottom_k=0,
        cooldown_months=int(preset.get("portfolio", {}).get("cooldown_months", 3)),
        min_track_months=int(preset.get("min_track_months", 24)),
        max_active=max(int(preset.get("selection_count", 10)) * 2, 50),
        max_weight=float(preset.get("portfolio", {}).get("max_weight", 0.15)),
        metrics=metrics,
    )


def _normalise_metric_weights(raw: Mapping[str, Any]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for key, value in raw.items():
        metric = UI_METRIC_ALIASES.get(str(key).lower())
        if metric is None:
            continue
        try:
            weight = float(value)
        except Exception:
            continue
        weights[metric] = weight
    total = sum(weights.values())
    if total <= 0:
        default = 1.0 / 3
        return {"sharpe": default, "return_ann": default, "drawdown": default}
    return {name: weight / total for name, weight in weights.items()}


def _build_pipeline_config(
    sim_config: Mapping[str, Any],
    metric_weights: Mapping[str, float],
    benchmark: str | None,
) -> Config:
    start = pd.Timestamp(sim_config["start"])
    end = pd.Timestamp(sim_config["end"])
    lookback = int(sim_config["lookback_months"])
    policy = sim_config["policy"]
    weighting_scheme = sim_config["portfolio"]["weighting_scheme"]

    blended_weights = {
        PIPELINE_METRIC_ALIASES.get(metric, metric): float(weight)
        for metric, weight in metric_weights.items()
    }

    registry = list(blended_weights.keys())

    sample_split = {
        "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
        "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
        "out_start": start.strftime("%Y-%m"),
        "out_end": end.strftime("%Y-%m"),
    }

    portfolio = {
        "selection_mode": "rank",
        "rank": {
            "inclusion_approach": "top_n",
            "n": int(policy.get("top_k", 5)),
            "score_by": "blended",
            "blended_weights": blended_weights,
        },
        "weighting_scheme": weighting_scheme,
    }
    benchmarks = {"SPX": benchmark} if benchmark else {}

    return Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={
            "target_vol": float(sim_config.get("risk_target", 0.1)),
            "floor_vol": 0.015,
            "warmup_periods": 0,
        },
        sample_split=sample_split,
        portfolio=portfolio,
        benchmarks=benchmarks,
        metrics={"registry": registry},
        export={},
        run={"monthly_cost": 0.0},
        seed=42,
    )


def _prepare_demo_setup(df: pd.DataFrame) -> DemoSetup:
    preset_data = _load_preset(DEFAULT_PRESET)
    metric_weights = _normalise_metric_weights(preset_data.get("metrics", {}))

    lookback = int(preset_data.get("lookback_months", 36))
    start, end = _derive_window(df, lookback)
    benchmark = _select_benchmark(df.columns)
    return_cols = [c for c in df.columns if c != benchmark]

    column_mapping = {
        "date_column": "Date",
        "return_columns": return_cols,
        "benchmark_column": benchmark,
        "risk_free_column": None,
        "column_display_names": {col: col for col in return_cols},
        "column_tickers": {},
    }

    policy = _build_policy(metric_weights, preset_data)

    overrides = {
        "lookback_months": lookback,
        "rebalance_frequency": preset_data.get("rebalance_frequency", "monthly"),
        "min_track_months": int(preset_data.get("min_track_months", 24)),
        "selection_count": int(preset_data.get("selection_count", 10)),
        "risk_target": float(preset_data.get("risk_target", 0.10)),
        "cooldown_months": policy.cooldown_months,
        "selected_metrics": list(metric_weights.keys()),
        "metric_weights": metric_weights,
        "weighting_scheme": "equal",
    }

    config_state = {
        "preset_name": DEFAULT_PRESET,
        "preset_config": preset_data,
        "column_mapping": column_mapping,
        "custom_overrides": overrides,
        "validation_errors": [],
        "is_valid": True,
    }

    sim_config = {
        "start": start,
        "end": end,
        "freq": overrides["rebalance_frequency"],
        "lookback_months": lookback,
        "benchmark": benchmark,
        "cash_rate": 0.0,
        "policy": policy.dict(),
        "rebalance": {
            "bayesian_only": True,
            "strategies": ["drift_band"],
            "params": {},
        },
        "risk_target": overrides["risk_target"],
        "column_mapping": column_mapping,
        "preset_name": DEFAULT_PRESET,
        "portfolio": {"weighting_scheme": overrides["weighting_scheme"]},
    }

    pipeline_config = _build_pipeline_config(sim_config, metric_weights, benchmark)
    return DemoSetup(config_state, sim_config, pipeline_config, benchmark)


def _update_session_state(
    st_module: Any, setup: DemoSetup, df: pd.DataFrame, meta: SchemaMeta
) -> None:
    state: MutableMapping[str, Any] = st_module.session_state
    state["returns_df"] = df
    state["schema_meta"] = meta
    state["benchmark_candidates"] = infer_benchmarks(list(df.columns))
    state["config_state"] = setup.config_state
    state["validation_messages"] = []
    state["sim_config"] = setup.sim_config
    state["demo_show_export_prompt"] = True
    state["demo_last_run"] = {
        "preset": DEFAULT_PRESET,
        "rows": df.shape[0],
        "cols": df.shape[1],
    }
    column_mapping = setup.config_state.get("column_mapping", {})
    if column_mapping:
        state["model_column_mapping"] = column_mapping
    overrides = setup.config_state.get("custom_overrides", {})
    trend_payload = {}
    preprocessing = getattr(setup.pipeline_config, "preprocessing", {}) or {}
    if isinstance(preprocessing, Mapping):
        trend_payload = preprocessing.get("trend", {}) or {}
    if setup.config_state.get("preset_name") and isinstance(trend_payload, dict):
        trend_payload = dict(trend_payload)
        trend_payload["preset"] = setup.config_state.get("preset_name")
    model_settings = ModelSettings(
        lookback_months=int(
            overrides.get(
                "lookback_months", setup.sim_config.get("lookback_months", 36)
            )
        ),
        rebalance_frequency=str(setup.sim_config.get("freq", "monthly")),
        selection_count=int(overrides.get("selection_count", 10)),
        risk_target=float(overrides.get("risk_target", 0.10)),
        weighting_scheme=str(
            setup.sim_config.get("portfolio", {}).get("weighting_scheme", "equal")
        ),
        cooldown_months=int(overrides.get("cooldown_months", 3)),
        min_track_months=int(overrides.get("min_track_months", 24)),
        metric_weights={
            k: float(v) for k, v in (overrides.get("metric_weights", {}) or {}).items()
        },
        trend_spec=trend_payload if isinstance(trend_payload, Mapping) else {},
        benchmark=setup.benchmark,
    )
    state["model_settings"] = model_settings


def run_one_click_demo(st_module: Any | None = None) -> bool:
    """Execute the demo pipeline and stash results in ``st.session_state``."""

    if st_module is None:
        import streamlit as st  # noqa: WPS433 - local import for testability

        st_module = st

    try:
        df, meta = _load_demo_returns()
    except Exception as exc:  # pragma: no cover - defensive guard
        st_module.error(f"Unable to load demo returns data: {exc}")
        return False

    try:
        setup = _prepare_demo_setup(df)
    except Exception as exc:  # pragma: no cover - unexpected config issues
        st_module.error(f"Failed to prepare demo configuration: {exc}")
        return False

    returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    try:
        result = run_simulation(setup.pipeline_config, returns)
    except Exception as exc:
        st_module.error(f"Demo simulation failed: {exc}")
        return False

    _update_session_state(st_module, setup, df, meta)
    st_module.session_state["sim_results"] = result
    return True
