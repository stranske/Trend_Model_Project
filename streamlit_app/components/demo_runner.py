"""Helpers for running the end-to-end demo pipeline from the Streamlit app."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import pandas as pd

from streamlit_app.components.data_schema import (
    SchemaMeta,
    infer_benchmarks,
    infer_risk_free_candidates,
    load_and_validate_file,
)
from streamlit_app.components.policy_engine import MetricSpec, PolicyConfig
from streamlit_app.components.universe_membership_input import (
    membership_cache_fingerprint,
)
from trend_analysis.api import run_simulation
from trend_analysis.config import Config
from trend_analysis.presets import get_trend_preset, list_trend_presets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "demo"

DEMO_DATA_CANDIDATES = (
    DEMO_DIR / "demo_returns.csv",
    DEMO_DIR / "demo_returns.xlsx",
)

DEFAULT_PRESET = "Balanced"
DEMO_PRESET_SELECTOR_LABEL = "Demo Settings Preset"
DEMO_PRESET_SELECTOR_HELP = (
    "Choose the built-in demo settings preset; model configuration presets live "
    "on the Model page."
)

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
ZERO_COST_MODEL = {"per_trade_bps": 0.0, "half_spread_bps": 0.0}


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
    raise FileNotFoundError("Demo returns not found. Expected demo/demo_returns.(csv|xlsx).")


def _load_preset(name: str) -> Dict[str, Any]:
    """Return a copy of the canonical full-config preset payload."""

    try:
        return get_trend_preset(name).config_mapping()
    except KeyError:
        return {}


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
    month_end = period.to_timestamp("M").as_unit("ns")
    return month_end.replace(hour=23, minute=59, second=59, microsecond=999999, nanosecond=999)


def _derive_window(
    df: pd.DataFrame, lookback_periods: int, oos_periods: int = 12
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = _month_end(pd.Timestamp(df.index.max()))
    start = _month_end(end - pd.DateOffset(months=max(oos_periods - 1, 0)))
    earliest = _month_end(pd.Timestamp(df.index.min()) + pd.DateOffset(months=lookback_periods))
    if start < earliest:
        start = earliest
    if start > end:
        start = end
    return start, end


def _build_policy(metric_weights: Mapping[str, float], preset: Mapping[str, Any]) -> PolicyConfig:
    metrics = [
        MetricSpec(name=metric, weight=float(weight)) for metric, weight in metric_weights.items()
    ]
    return PolicyConfig(
        top_k=int(preset.get("selection_count", 10)),
        bottom_k=0,
        cooldown_months=int(preset.get("portfolio", {}).get("cooldown_periods", 3)),
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
    lookback = int(sim_config["lookback_periods"])
    policy = sim_config["policy"]
    weighting_name = sim_config["portfolio"]["weighting"]["name"]

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
        "indices_list": [benchmark] if benchmark else [],
        "selection_mode": "rank",
        "rank": {
            "inclusion_approach": "top_n",
            "n": int(policy.get("top_k", 5)),
            "score_by": "blended",
            "blended_weights": blended_weights,
        },
        "weighting": {"name": weighting_name, "params": {}},
        "cost_model": dict(ZERO_COST_MODEL),
    }
    benchmarks = {"SPX": benchmark} if benchmark else {}

    return Config(
        version="1",
        data={
            "allow_risk_free_fallback": True,  # Enable auto-detection of risk-free column
        },
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

    lookback = int(preset_data.get("lookback_periods", 36))
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
        "lookback_periods": lookback,
        "rebalance_frequency": preset_data.get("rebalance_frequency", "monthly"),
        "min_track_months": int(preset_data.get("min_track_months", 24)),
        "selection_count": int(preset_data.get("selection_count", 10)),
        "risk_target": float(preset_data.get("risk_target", 0.10)),
        "cooldown_periods": policy.cooldown_months,
        "selected_metrics": list(metric_weights.keys()),
        "metric_weights": metric_weights,
        "weighting_name": "equal",
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
        "lookback_periods": lookback,
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
        "portfolio": {
            "weighting": {"name": overrides["weighting_name"], "params": {}},
            "cost_model": dict(ZERO_COST_MODEL),
        },
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
    # Build the current model-settings snapshot consumed by the results page.
    lookback = int(overrides.get("lookback_periods", setup.sim_config.get("lookback_periods", 36)))
    selection_count = int(overrides.get("selection_count", 10))
    risk_target = float(overrides.get("risk_target", 0.10))
    weighting_cfg = setup.sim_config.get("portfolio", {}).get("weighting", {})
    weighting_name = str(weighting_cfg.get("name", "equal"))
    metric_weights_dict = {
        k: float(v) for k, v in (overrides.get("metric_weights", {}) or {}).items()
    }

    # Set the current model-state snapshot consumed by the Results page.
    state["model_state"] = {
        "preset": setup.config_state.get("preset_name", "Baseline"),
        "trend_spec": trend_payload if isinstance(trend_payload, Mapping) else {},
        "lookback_periods": lookback,
        "min_history_periods": lookback,
        "evaluation_periods": 12,
        "selection_count": selection_count,
        "weighting_name": weighting_name,
        "metric_weights": (
            metric_weights_dict
            if metric_weights_dict
            else {
                "sharpe": 1.0,
                "return_ann": 1.0,
                "drawdown": 0.5,
                "sortino": 0.0,
                "info_ratio": 0.0,
                "vol": 0.0,
            }
        ),
        "risk_target": risk_target,
        "warmup_periods": 0,
        "info_ratio_benchmark": setup.benchmark or "",
    }
    state["selected_benchmark"] = setup.benchmark


def _selected_funds_from_result(result: Any, df: pd.DataFrame, benchmark: str | None) -> list[str]:
    selected: list[str] = []
    for attr in ("weights", "exposures", "portfolio"):
        series = getattr(result, attr, None)
        if isinstance(series, pd.Series):
            selected.extend(str(item) for item in series.index if str(item) in df.columns)
    if not selected:
        selected = [str(col) for col in df.columns]

    excluded = {benchmark, None, ""}
    return [fund for fund in dict.fromkeys(selected) if fund not in excluded]


def _selected_risk_free_from_demo(df: pd.DataFrame, benchmark: str | None) -> str | None:
    for candidate in infer_risk_free_candidates(list(df.columns)):
        if candidate != benchmark:
            return candidate
    return None


def _analysis_run_key(
    state: Mapping[str, Any], model_state: Mapping[str, Any], benchmark: str | None
) -> str:
    fingerprint = state.get("data_fingerprint", "unknown")
    model_blob = json.dumps(model_state, sort_keys=True, default=str)
    bench = benchmark or "__none__"
    selected_rf = state.get("selected_risk_free")
    selected_rf_key = selected_rf or "__none__"
    applied_funds = state.get("analysis_fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = state.get("fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = []

    info_ratio_benchmark = model_state.get("info_ratio_benchmark")
    prohibited = {selected_rf, benchmark, info_ratio_benchmark} - {None}
    sanitized_funds = [c for c in applied_funds if c not in prohibited]
    funds_blob = json.dumps(list(sanitized_funds), sort_keys=False, default=str)
    funds_hash = hashlib.sha256(funds_blob.encode("utf-8")).hexdigest()[:12]
    membership_hash = hashlib.sha256(
        membership_cache_fingerprint(state).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{fingerprint}:{bench}:{selected_rf_key}:{funds_hash}:" f"{membership_hash}:{model_blob}"
    )


def _store_demo_result_state(
    st_module: Any, setup: DemoSetup, df: pd.DataFrame, result: Any
) -> None:
    from streamlit_app.components.data_cache import cache_key_for_frame

    state: MutableMapping[str, Any] = st_module.session_state
    selected_rf = _selected_risk_free_from_demo(df, setup.benchmark)
    state["selected_risk_free"] = selected_rf
    selected_funds = _selected_funds_from_result(result, df, setup.benchmark)
    state["selected_fund_columns"] = list(selected_funds)
    state["fund_columns"] = list(selected_funds)
    state["analysis_fund_columns"] = list(selected_funds)
    state["sim_results"] = result
    state["analysis_result"] = result
    state["data_fingerprint"] = cache_key_for_frame(df)
    model_state = state.get("model_state")
    if isinstance(model_state, Mapping):
        state["analysis_result_key"] = _analysis_run_key(state, model_state, setup.benchmark)


def run_one_click_demo(st_module: Any | None = None) -> bool:
    """Execute the demo pipeline and stash results in ``st.session_state``."""

    if st_module is None:
        import streamlit as st  # noqa: E402 - local import for testability

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
    _store_demo_result_state(st_module, setup, df, result)
    return True


def list_presets() -> list[Dict[str, Any]]:
    """Return a list of available preset configurations."""
    return [
        {
            "name": preset.label,
            "description": preset.description,
            "file": f"{preset.slug}.yml",
        }
        for preset in list_trend_presets()
    ]


def load_preset_config(name: str) -> Dict[str, Any]:
    """Load a preset configuration by name."""
    return _load_preset(name)


def run_demo_with_overrides(
    preset_name: str = "Balanced",
    overrides: Dict[str, Any] | None = None,
    st_module: Any | None = None,
) -> bool:
    """Execute the demo pipeline with user-specified overrides.

    Parameters
    ----------
    preset_name
        Name of the preset to use as base configuration.
    overrides
        Dictionary of parameter overrides (lookback_periods, selection_count, etc.)
    st_module
        Streamlit module for error display (defaults to st).

    Returns
    -------
    bool
        True if demo completed successfully.
    """
    if st_module is None:
        import streamlit as st  # noqa: E402

        st_module = st

    overrides = overrides or {}

    # Load demo data
    try:
        df, meta = _load_demo_returns()
    except Exception as exc:
        st_module.error(f"Unable to load demo returns data: {exc}")
        return False

    # Load preset and apply overrides
    preset_data = _load_preset(preset_name)

    # Deep merge overrides into preset
    merged = dict(preset_data)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    # Normalize metric weights
    raw_metrics = merged.get("metrics", {})
    metric_weights = _normalise_metric_weights(raw_metrics)

    # Derive time window
    lookback = int(merged.get("lookback_periods", 36))
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

    # Build policy with merged settings
    policy = _build_policy(metric_weights, merged)

    # Update policy with additional overrides
    portfolio_overrides = merged.get("portfolio", {})
    if "max_weight" in portfolio_overrides:
        policy = PolicyConfig(
            top_k=policy.top_k,
            bottom_k=policy.bottom_k,
            cooldown_months=int(
                portfolio_overrides.get("cooldown_periods", policy.cooldown_months)
            ),
            min_track_months=policy.min_track_months,
            max_active=policy.max_active,
            max_weight=float(portfolio_overrides.get("max_weight", policy.max_weight)),
            metrics=policy.metrics,
        )

    # Build config state
    config_state = {
        "preset_name": preset_name,
        "preset_config": merged,
        "column_mapping": column_mapping,
        "custom_overrides": merged,
        "validation_errors": [],
        "is_valid": True,
    }

    sim_config = {
        "start": start,
        "end": end,
        "freq": merged.get("rebalance_frequency", "monthly"),
        "lookback_periods": lookback,
        "benchmark": benchmark,
        "cash_rate": 0.0,
        "policy": policy.dict(),
        "rebalance": {
            "bayesian_only": True,
            "strategies": ["drift_band"],
            "params": {},
        },
        "risk_target": float(merged.get("risk_target", 0.10)),
        "column_mapping": column_mapping,
        "preset_name": preset_name,
        "portfolio": {
            "weighting": {"name": merged.get("weighting_name", "equal"), "params": {}},
            "cost_model": dict(ZERO_COST_MODEL),
        },
    }

    pipeline_config = _build_pipeline_config(sim_config, metric_weights, benchmark)

    setup = DemoSetup(config_state, sim_config, pipeline_config, benchmark)

    # Run simulation
    returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    try:
        result = run_simulation(setup.pipeline_config, returns)
    except Exception as exc:
        st_module.error(f"Demo simulation failed: {exc}")
        return False

    # Update session state
    _update_session_state(st_module, setup, df, meta)
    _store_demo_result_state(st_module, setup, df, result)

    return True
