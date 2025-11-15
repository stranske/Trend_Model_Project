"""Configure page for Streamlit trend analysis app."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


def initialize_config_state():
    """Initialize configuration in session state if not present."""
    if "sim_config" not in st.session_state:
        st.session_state["sim_config"] = {
            "start": date(2020, 1, 1),
            "end": date(2023, 12, 31),
            "lookback_months": 12,
            "risk_target": 1.0,
            "portfolio": {},
            "benchmarks": {},
            "metrics": {},
            "run": {},
        }


def validate_date_range(start_date, end_date, lookback_months):
    """Validate that the date range makes sense."""
    issues = []

    if start_date >= end_date:
        issues.append("Start date must be before end date")

    # Check if we have enough data for lookback
    min_start = pd.to_datetime(start_date) - pd.DateOffset(months=lookback_months)
    if "returns_df" in st.session_state and st.session_state["returns_df"] is not None:
        df = st.session_state["returns_df"]
        if "Date" in df.columns:
            data_start = pd.to_datetime(df["Date"]).min()
            if data_start > min_start:
                issues.append(
                    f"Insufficient data for {lookback_months}-month lookback. Data starts {data_start.date()}"
                )

    return issues


def main():
    """Main function for the Configure page."""
    st.title("⚙️ Configure Analysis")

    # Initialize state
    initialize_config_state()

    # Check if data is uploaded
    if "returns_df" not in st.session_state or st.session_state["returns_df"] is None:
        st.warning("⚠️ Please upload your data first before configuring the analysis.")
        st.info("👈 Go to the **Upload** page to load your data.")
        return

    df = st.session_state["returns_df"]
    st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Configuration sections
    config = st.session_state["sim_config"]

    # Date Range Configuration
    st.markdown("### 📅 Analysis Period")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date", value=config["start"], help="Beginning of the analysis period"
        )

    with col2:
        end_date = st.date_input(
            "End Date", value=config["end"], help="End of the analysis period"
        )

    lookback_months = st.slider(
        "Lookback Period (months)",
        min_value=1,
        max_value=60,
        value=config["lookback_months"],
        help="Number of months to look back for training data",
    )

    # Validate dates
    date_issues = validate_date_range(start_date, end_date, lookback_months)
    if date_issues:
        for issue in date_issues:
            st.error(f"❌ {issue}")
    else:
        st.success("✅ Date range is valid")

    # Risk Configuration
    st.markdown("### 🎯 Risk Management")

    risk_target = st.number_input(
        "Target Volatility",
        min_value=0.1,
        max_value=5.0,
        value=config["risk_target"],
        step=0.1,
        help="Target annual volatility for portfolio (1.0 = 100%)",
    )

    # Portfolio Configuration
    st.markdown("### 📊 Portfolio Settings")

    selection_mode = st.selectbox(
        "Selection Mode",
        ["all", "top_k", "random", "custom"],
        index=0,
        help="How to select assets for the portfolio",
    )

    if selection_mode == "top_k":
        top_k = st.number_input(
            "Number of Top Assets", min_value=1, max_value=50, value=10
        )
        config["portfolio"]["top_k"] = top_k

    elif selection_mode == "random":
        random_n = st.number_input(
            "Number of Random Assets", min_value=1, max_value=50, value=8
        )
        config["portfolio"]["random_n"] = random_n
        config["portfolio"]["random_seed"] = st.number_input("Random Seed", value=42)

    config["portfolio"]["selection_mode"] = selection_mode

    # Advanced Settings
    with st.expander("🔧 Advanced Settings", expanded=False):

        st.markdown("#### Transaction Costs")
        monthly_cost = st.number_input(
            "Monthly Transaction Cost (%)",
            min_value=0.0,
            max_value=10.0,
            value=config["run"].get("monthly_cost", 0.0),
            step=0.01,
            help="Monthly transaction cost as percentage",
        )
        config["run"]["monthly_cost"] = monthly_cost

        st.markdown("#### Metrics")
        enable_metrics = st.checkbox("Enable Advanced Metrics", value=False)
        if enable_metrics:
            available_metrics = ["sharpe", "sortino", "calmar", "max_drawdown", "cagr"]
            selected_metrics = st.multiselect(
                "Select Metrics to Calculate",
                available_metrics,
                default=["sharpe", "max_drawdown"],
            )
            config["metrics"]["registry"] = selected_metrics
        else:
            config["metrics"] = {}

    # Update config in session state
    config.update(
        {
            "start": start_date,
            "end": end_date,
            "lookback_months": lookback_months,
            "risk_target": risk_target,
        }
    )

    st.session_state["sim_config"] = config

    # Configuration Summary
    st.markdown("---")
    st.markdown("### 📋 Configuration Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"**Analysis Period**\n"
            f"- Start: {start_date}\n"
            f"- End: {end_date}\n"
            f"- Lookback: {lookback_months} months"
        )

    with col2:
        st.info(
            f"**Settings**\n"
            f"- Target Vol: {risk_target:.1%}\n"
            f"- Selection: {selection_mode}\n"
            f"- Transaction Cost: {config['run'].get('monthly_cost', 0.0):.2%}/month"
        )

    # Save/Load Configuration
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            st.success("✅ Configuration saved!")

    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.session_state["sim_config"] = {
                "start": date(2020, 1, 1),
                "end": date(2023, 12, 31),
                "lookback_months": 12,
                "risk_target": 1.0,
                "portfolio": {},
                "benchmarks": {},
                "metrics": {},
                "run": {},
            }
            st.success("🔄 Configuration reset to defaults!")
            st.rerun()

    with col3:
        if st.button("▶️ Continue to Run"):
            st.info("👉 Go to the **Run** page to execute your analysis!")

    # Show raw config for debugging
    if st.checkbox("🔍 Show Raw Configuration", value=False):
        st.json(config)


if __name__ == "__main__":
    main()
