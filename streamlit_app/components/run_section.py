"""Run controls embedded on the Model page."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Mapping

import pandas as pd
import streamlit as st

from streamlit_app.components.disclaimer import show_disclaimer
from streamlit_app.components.guardrails import (
    estimate_resource_usage,
    prepare_dry_run_plan,
)
from trend_analysis.api import run_simulation as run_simulation_impl

RunSimulationFn = Callable[["StreamlitConfig", pd.DataFrame], Any]


class StreamlitConfig:
    """Thin wrapper to mimic the dataclass used by the CLI pipeline."""

    def __init__(self, **data: object) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _coerce_mapping(value: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _resolve_run_simulation() -> RunSimulationFn:
    return run_simulation_impl


def _normalise_signals_config(signals: Mapping[str, object] | object) -> Mapping[str, object]:
    mapping = _coerce_mapping(signals)
    if not mapping:
        return {}
    return {key: value for key, value in mapping.items() if value is not None}


def render_run_section(config_state: Mapping[str, object]) -> None:
    """Render the execution controls and run the simulation when requested."""

    if not st.session_state.get("returns_df"):
        st.info("Upload data before running a simulation.")
        return

    if not config_state.get("is_valid"):
        st.info("Save a valid configuration to unlock the run controls.")
        return

    df = st.session_state.get("returns_df")
    config_dict = config_state.get("custom_overrides", {})
    accepted = show_disclaimer()
    col_dry, col_run = st.columns([1, 1])
    with col_dry:
        dry_run_clicked = st.button(
            "Dry run sample", disabled=not accepted, help="Use a small slice to sanity-check."
        )
    with col_run:
        run_clicked = st.button(
            "Run full simulation",
            type="primary",
            disabled=not accepted,
        )

    if not (dry_run_clicked or run_clicked):
        return

    if df is None:
        st.error("Missing data frame in session state.")
        return

    config_overrides = _coerce_mapping(config_dict)
    run_sim = _resolve_run_simulation()

    returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    start = config_overrides.get("start")
    end = config_overrides.get("end")
    lookback = config_overrides.get("lookback_months", 0)
    try:
        lookback = int(lookback) if lookback is not None else 0
    except Exception:
        lookback = 0

    try:
        start_ts = pd.to_datetime(start) if start is not None else None
        end_ts = pd.to_datetime(end) if end is not None else None
    except Exception:
        start_ts = None
        end_ts = None

    if start_ts is None or end_ts is None:
        st.error("Configuration is missing start and end dates.")
        return

    signals_cfg = _normalise_signals_config(config_state.get("signals", {}))
    portfolio_cfg = {
        "weighting_scheme": config_overrides.get("weighting_scheme", "equal"),
    }

    config_data: Dict[str, object] = {
        "version": "1",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {"target_vol": config_overrides.get("risk_target", 0.1)},
        "signals": signals_cfg,
        "sample_split": {
            "in_start": (start_ts - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
            "in_end": (start_ts - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "out_start": start_ts.strftime("%Y-%m"),
            "out_end": end_ts.strftime("%Y-%m"),
        },
        "portfolio": portfolio_cfg,
        "metrics": {},
        "export": {},
        "benchmarks": {},
        "run": {"trend_preset": config_state.get("trend_preset")},
    }

    # Guardrail estimates
    estimate = config_state.get("resource_estimate")
    if not estimate:
        estimate = estimate_resource_usage(df.shape[0], df.shape[1])
        st.session_state.config_state["resource_estimate"] = estimate
    st.caption(
        f"Approximate resources: {estimate.approx_memory_mb:.1f} MB memory, "
        f"{estimate.estimated_runtime_s/60:.1f} minutes runtime."
    )
    for warn in getattr(estimate, "warnings", ()):
        st.warning(warn)

    if dry_run_clicked and not run_clicked:
        try:
            plan = prepare_dry_run_plan(df, lookback or 0)
        except ValueError as exc:
            st.error(f"Dry run unavailable: {exc}")
            return
        dry_returns = plan.frame.reset_index().rename(
            columns={plan.frame.index.name or "index": "Date"}
        )
        dry_run_id = f"dry-{uuid.uuid4().hex[:10]}"
        dry_config = StreamlitConfig(
            **config_data,
            sample_split=plan.sample_split(),
            run_id=dry_run_id,
        )
        with st.spinner("Running dry run on sample slice..."):
            result = run_sim(dry_config, dry_returns)
        st.session_state["dry_run_results"] = result
        st.session_state["dry_run_summary"] = plan.summary()
        st.success(
            f"Dry run complete on {plan.frame.shape[0]} rows × {plan.frame.shape[1]} columns."
        )
        st.json(plan.summary())
        return

    run_id = f"wf-{uuid.uuid4().hex[:10]}"
    config_payload = StreamlitConfig(**config_data, run_id=run_id)
    with st.spinner("Running full simulation… this can take a few minutes"):
        result = run_sim(config_payload, returns)
    st.session_state["sim_results"] = result
    st.session_state["sim_config"] = config_payload.model_dump()
    st.session_state["run_id"] = run_id
    st.success("Simulation completed! Jump to the Results page to review outputs.")


__all__ = ["render_run_section"]
