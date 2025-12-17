# --- Streamlit UI ---
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

import streamlit as st  # noqa: E402

from streamlit_app.components.demo_runner import (  # noqa: E402
    list_presets,
    load_preset_config,
    run_demo_with_overrides,
)

st.set_page_config(
    page_title="Portfolio Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Portfolio Simulator")
st.markdown(
    """
Welcome! This app analyzes trend-following fund portfolios with volatility adjustment.

**Quick Start:**
- Use the **Demo** section below to run analysis on sample data with preset configurations
- Use the **Custom Analysis** section to load your own data and configure every parameter

The demo uses a specialized policy-based engine optimized for the sample dataset.
For full control over all parameters, use the Custom Analysis flow.
    """
)

st.markdown("---")

# =============================================================================
# DEMO SECTION
# =============================================================================
st.subheader("🎯 Demo with Sample Data")

# Get available presets
presets = list_presets()
preset_names = [p["name"] for p in presets]

# Initialize demo settings in session state
if "demo_preset" not in st.session_state:
    st.session_state["demo_preset"] = "Balanced"
if "demo_settings_expanded" not in st.session_state:
    st.session_state["demo_settings_expanded"] = False

# Preset selector
selected_preset = st.selectbox(
    "Strategy Preset",
    preset_names,
    index=(
        preset_names.index(st.session_state["demo_preset"])
        if st.session_state["demo_preset"] in preset_names
        else 0
    ),
    key="demo_preset_selector",
    help="Choose a pre-configured strategy profile",
)
st.session_state["demo_preset"] = selected_preset

# Load preset config for display and modification
preset_config = load_preset_config(selected_preset)

# Show preset description
if preset_config.get("description"):
    st.caption(f"*{preset_config.get('description')}*")

# Expandable settings section
with st.expander(
    "⚙️ Customize Demo Settings",
    expanded=st.session_state.get("demo_settings_expanded", False),
):
    st.markdown("Adjust key parameters before running the demo:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Time & Selection**")
        demo_lookback = st.number_input(
            "Lookback Periods",
            min_value=6,
            max_value=120,
            value=int(preset_config.get("lookback_periods", 36)),
            key="demo_lookback",
            help="Historical period for calculating metrics",
        )
        demo_selection_count = st.number_input(
            "Selection Count (top_k)",
            min_value=1,
            max_value=50,
            value=int(preset_config.get("selection_count", 10)),
            key="demo_selection_count",
            help="Number of top-ranked funds to include per rebalance",
        )
        demo_min_track = st.number_input(
            "Min Track Record (months)",
            min_value=6,
            max_value=60,
            value=int(preset_config.get("min_track_months", 24)),
            key="demo_min_track",
            help="Minimum history required for a fund to be eligible",
        )

    with col2:
        st.markdown("**Risk & Portfolio**")
        demo_risk_target = st.number_input(
            "Target Volatility",
            min_value=0.01,
            max_value=0.50,
            value=float(preset_config.get("risk_target", 0.10)),
            format="%.2f",
            key="demo_risk_target",
            help="Target annualized volatility for the portfolio",
        )
        demo_max_weight = st.number_input(
            "Max Position Weight",
            min_value=0.01,
            max_value=1.0,
            value=float(preset_config.get("portfolio", {}).get("max_weight", 0.15)),
            format="%.2f",
            key="demo_max_weight",
            help="Maximum weight for any single position",
        )
        demo_cooldown = st.number_input(
            "Cooldown Months",
            min_value=0,
            max_value=24,
            value=int(preset_config.get("portfolio", {}).get("cooldown_months", 3)),
            key="demo_cooldown",
            help="Periods before a dropped fund can be re-added",
        )

    st.markdown("**Metric Weights** (blended scoring)")
    metric_cols = st.columns(4)
    preset_metrics = preset_config.get("metrics", {})

    with metric_cols[0]:
        demo_w_sharpe = st.number_input(
            "Sharpe",
            min_value=0.0,
            max_value=1.0,
            value=float(preset_metrics.get("sharpe_ratio", 0.3)),
            format="%.2f",
            key="demo_w_sharpe",
        )
    with metric_cols[1]:
        demo_w_return = st.number_input(
            "Return",
            min_value=0.0,
            max_value=1.0,
            value=float(preset_metrics.get("return_ann", 0.3)),
            format="%.2f",
            key="demo_w_return",
        )
    with metric_cols[2]:
        demo_w_drawdown = st.number_input(
            "Max DD",
            min_value=0.0,
            max_value=1.0,
            value=float(preset_metrics.get("max_drawdown", 0.25)),
            format="%.2f",
            key="demo_w_drawdown",
        )
    with metric_cols[3]:
        demo_w_vol = st.number_input(
            "Volatility",
            min_value=0.0,
            max_value=1.0,
            value=float(preset_metrics.get("volatility", 0.15)),
            format="%.2f",
            key="demo_w_vol",
        )

    st.caption("Weights will be normalized automatically.")

# Run Demo button
if st.button("🚀 Run Demo", type="primary", use_container_width=True):
    # Collect overrides from session state
    overrides = {
        "lookback_periods": st.session_state.get("demo_lookback", 36),
        "selection_count": st.session_state.get("demo_selection_count", 10),
        "min_track_months": st.session_state.get("demo_min_track", 24),
        "risk_target": st.session_state.get("demo_risk_target", 0.10),
        "portfolio": {
            "max_weight": st.session_state.get("demo_max_weight", 0.15),
            "cooldown_months": st.session_state.get("demo_cooldown", 3),
        },
        "metrics": {
            "sharpe_ratio": st.session_state.get("demo_w_sharpe", 0.3),
            "return_ann": st.session_state.get("demo_w_return", 0.3),
            "max_drawdown": st.session_state.get("demo_w_drawdown", 0.25),
            "volatility": st.session_state.get("demo_w_vol", 0.15),
        },
    }

    with st.spinner("Loading demo data and running analysis..."):
        success = run_demo_with_overrides(
            preset_name=st.session_state.get("demo_preset", "Balanced"),
            overrides=overrides,
        )

    if success:
        st.success("Demo complete! Viewing results...")
        st.switch_page("pages/3_Results.py")
    else:
        st.error("Demo failed. Check the error message above.")

st.markdown("---")

# =============================================================================
# CUSTOM ANALYSIS SECTION
# =============================================================================
st.subheader("🔧 Custom Analysis")
st.markdown(
    """
    For full control over all parameters, use the manual workflow:
    1. **Data** - Load your own CSV/Excel file
    2. **Model** - Configure all analysis parameters
    3. **Results** - View and export results
    """
)

if st.button("📂 Go to Data Upload", use_container_width=True):
    st.switch_page("pages/1_Data.py")

st.caption(
    "The Model page uses a different analysis pipeline with more configuration options."
)
