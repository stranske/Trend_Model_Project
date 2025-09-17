"""Helpers for the Streamlit one-click demo run."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import yaml

from trend_portfolio_app.data_schema import infer_benchmarks, load_and_validate_file
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import Simulator

DEMO_FILENAMES: Sequence[str] = ("demo_returns.csv", "demo_returns.xlsx")
DEFAULT_PRESET_NAME = "Balanced"

METRIC_NAME_MAP: Mapping[str, str] = {
    "sharpe_ratio": "sharpe",
    "sharpe": "sharpe",
    "return_ann": "return_ann",
    "annual_return": "return_ann",
    "max_drawdown": "drawdown",
    "drawdown": "drawdown",
    "volatility": "vol",
    "vol": "vol",
}

NAV_QUEUE_KEY = "demo_nav_queue"
NAV_READY_FLAG = "demo_nav_ready"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_demo_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the bundled demo dataset and validate its schema."""
    demo_dir = _repo_root() / "demo"
    for name in DEMO_FILENAMES:
        path = demo_dir / name
        if path.exists():
            with path.open("rb") as handle:
                df, meta = load_and_validate_file(handle)
            return df, meta
    raise FileNotFoundError("demo dataset not found in ./demo")


def _preset_path(name: str) -> Path:
    presets_dir = _repo_root() / "config" / "presets"
    return presets_dir / f"{name.lower()}.yml"


def load_preset_config(name: str) -> Dict[str, Any]:
    """Load a preset configuration from ``config/presets``."""
    path = _preset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"preset '{name}' not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"preset '{name}' is not a mapping")
    return data


def _convert_preset_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for raw_name, weight in metrics.items():
        norm = METRIC_NAME_MAP.get(str(raw_name).lower())
        if norm is None:
            continue
        try:
            value = float(weight)
        except Exception:
            continue
        converted[norm] = converted.get(norm, 0.0) + value
    if not converted or sum(converted.values()) <= 0:
        return {"sharpe": 1.0}
    total = sum(converted.values())
    return {k: v / total for k, v in converted.items()}


def build_demo_configuration(
    df: pd.DataFrame,
    preset: Mapping[str, Any],
    preset_name: str,
) -> Tuple[Dict[str, Any], PolicyConfig, Dict[str, Any], Dict[str, Any], Optional[str], List[str]]:
    """Create the configuration dictionary and policy for the demo run."""
    lookback = int(preset.get("lookback_months", 36))
    freq = str(preset.get("rebalance_frequency", "monthly"))
    min_track = int(preset.get("min_track_months", 24))
    selection_count = int(preset.get("selection_count", 10))
    risk_target = float(preset.get("risk_target", 0.10))

    metrics_map = _convert_preset_metrics(preset.get("metrics", {}))
    metric_specs = [MetricSpec(name=k, weight=float(v)) for k, v in metrics_map.items()]

    portfolio_cfg = preset.get("portfolio", {})
    cooldown = int(portfolio_cfg.get("cooldown_months", 3))
    max_weight = float(portfolio_cfg.get("max_weight", 0.10))

    candidates = infer_benchmarks(list(df.columns))
    benchmark = candidates[0] if candidates else None
    return_cols = [c for c in df.columns if c != benchmark]

    column_mapping = {
        "date_column": "Date",
        "return_columns": return_cols,
        "benchmark_column": benchmark,
        "risk_free_column": None,
        "column_display_names": {c: c for c in return_cols},
        "column_tickers": {},
    }

    overrides = {
        "lookback_months": lookback,
        "rebalance_frequency": freq,
        "min_track_months": min_track,
        "selection_count": selection_count,
        "risk_target": risk_target,
        "cooldown_months": cooldown,
        "selected_metrics": list(metrics_map.keys()),
        "metric_weights": metrics_map,
        "weighting_scheme": "equal",
    }

    policy = PolicyConfig(
        top_k=selection_count,
        bottom_k=0,
        cooldown_months=cooldown,
        min_track_months=min_track,
        max_active=100,
        max_weight=max_weight,
        metrics=metric_specs,
    )

    config_dict: Dict[str, Any] = {
        "start": pd.Timestamp(df.index.min()),
        "end": pd.Timestamp(df.index.max()),
        "freq": freq,
        "lookback_months": lookback,
        "benchmark": benchmark,
        "cash_rate": 0.0,
        "policy": policy.dict(),
        "rebalance": {
            "bayesian_only": True,
            "strategies": ["drift_band"],
            "params": {},
        },
        "risk_target": risk_target,
        "column_mapping": column_mapping,
        "preset_name": preset_name,
        "portfolio": {"weighting_scheme": overrides["weighting_scheme"]},
    }

    return config_dict, policy, column_mapping, overrides, benchmark, list(candidates)


def _execute_demo_simulation(
    df: pd.DataFrame,
    config_dict: Mapping[str, Any],
    policy: PolicyConfig,
    benchmark: Optional[str],
):
    simulator = Simulator(
        df,
        benchmark_col=benchmark,
        cash_rate=float(config_dict.get("cash_rate", 0.0)),
    )
    start = pd.Timestamp(config_dict.get("start"))
    end = pd.Timestamp(config_dict.get("end"))
    return simulator.run(
        start=start,
        end=end,
        freq=str(config_dict.get("freq", "monthly")),
        lookback_months=int(config_dict.get("lookback_months", 36)),
        policy=policy,
        rebalance=dict(config_dict.get("rebalance", {})),
    )


def _spinner_context(st_module, text: str):
    spinner = getattr(st_module, "spinner", None)
    if callable(spinner):
        return spinner(text)
    return nullcontext()


def enqueue_navigation(st_module, pages: Sequence[str]) -> None:
    st_module.session_state[NAV_QUEUE_KEY] = list(pages)
    st_module.session_state[NAV_READY_FLAG] = False


def safe_switch_page(st_module, page: str) -> bool:
    switch = getattr(st_module, "switch_page", None)
    if callable(switch):
        try:
            switch(page)
        except Exception:  # pragma: no cover - defensive
            return False
        return True
    return False


def handle_demo_navigation(st_module, current_page: str) -> Optional[str]:
    queue = st_module.session_state.get(NAV_QUEUE_KEY)
    if not isinstance(queue, list) or not queue:
        st_module.session_state[NAV_QUEUE_KEY] = []
        st_module.session_state.pop(NAV_READY_FLAG, None)
        return None

    queue = list(queue)
    if queue and queue[0] == current_page:
        queue.pop(0)

    if not queue:
        st_module.session_state[NAV_QUEUE_KEY] = []
        st_module.session_state.pop(NAV_READY_FLAG, None)
        return None

    if not st_module.session_state.get(NAV_READY_FLAG):
        st_module.session_state[NAV_QUEUE_KEY] = queue
        st_module.session_state[NAV_READY_FLAG] = True
        return None

    next_page = queue.pop(0)
    st_module.session_state[NAV_QUEUE_KEY] = queue
    st_module.session_state[NAV_READY_FLAG] = False

    if safe_switch_page(st_module, next_page):
        if not queue:
            st_module.session_state.pop(NAV_READY_FLAG, None)
        return next_page

    st_module.session_state[NAV_QUEUE_KEY] = []
    st_module.session_state.pop(NAV_READY_FLAG, None)
    return None


def run_one_click_demo(
    *,
    st_module=st,
    preset_name: str = DEFAULT_PRESET_NAME,
) -> None:
    """Load demo data, execute the simulation, and queue navigation."""
    try:
        with _spinner_context(st_module, "Running demo analysis..."):
            df, meta = load_demo_dataset()
            preset = load_preset_config(preset_name)
            config_dict, policy, mapping, overrides, benchmark, candidates = build_demo_configuration(
                df, preset, preset_name
            )

            st_module.session_state["returns_df"] = df
            st_module.session_state["schema_meta"] = meta
            st_module.session_state["benchmark_candidates"] = candidates
            st_module.session_state["sim_config"] = config_dict
            st_module.session_state["config_state"] = {
                "preset_name": preset_name,
                "preset_config": preset,
                "column_mapping": mapping,
                "custom_overrides": overrides,
                "validation_errors": [],
                "is_valid": True,
            }
            st_module.session_state["validation_messages"] = []

            result = _execute_demo_simulation(df, config_dict, policy, benchmark)
            st_module.session_state["sim_results"] = result
    except FileNotFoundError as err:
        st_module.error(f"Demo data unavailable: {err}")
        return
    except Exception as exc:  # pragma: no cover - safety net for unexpected issues
        st_module.error(f"Demo run failed: {exc}")
        return

    st_module.success("Demo run complete! Results and export are ready.")
    enqueue_navigation(st_module, ["pages/4_Results.py", "pages/5_Export.py"])
    safe_switch_page(st_module, "pages/4_Results.py")
