"""Settings Validation page for systematically testing UI settings.

This page allows interactive testing of individual settings to verify
they are properly wired into the analysis pipeline. Each test:
1. Runs a baseline configuration
2. Runs a variant with one setting changed
3. Shows side-by-side comparison of key metrics
4. Validates the effect matches economic intuition
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.components import analysis_runner

# =============================================================================
# Setting Definitions
# =============================================================================


@dataclass
class SettingDef:
    """Definition of a testable setting."""

    key: str  # Key in model_state
    label: str  # Human-readable label
    category: str  # Category for grouping
    baseline: Any  # Default/baseline value
    test_values: list[Any]  # Alternative values to test
    expected_effect: str  # What should change
    economic_intuition: str  # Why it should change that way


TESTABLE_SETTINGS: list[SettingDef] = [
    # === Core Selection ===
    SettingDef(
        key="selection_count",
        label="Selection Count",
        category="Selection",
        baseline=10,
        test_values=[5, 15, 20],
        expected_effect="Number of funds selected",
        economic_intuition="More selection count → more funds in portfolio",
    ),
    SettingDef(
        key="inclusion_approach",
        label="Selection Approach",
        category="Selection",
        baseline="threshold",
        test_values=["top_n", "top_pct", "random", "buy_and_hold"],
        expected_effect="Selection method and fund composition",
        economic_intuition="Different methods select different funds",
    ),
    SettingDef(
        key="rank_pct",
        label="Rank Percentage",
        category="Selection",
        baseline=0.10,
        test_values=[0.05, 0.20, 0.30],
        expected_effect="Number of funds (in top_pct mode)",
        economic_intuition="Higher % → more funds selected",
    ),
    # === Weighting ===
    SettingDef(
        key="weighting_scheme",
        label="Weighting Scheme",
        category="Weighting",
        baseline="equal",
        test_values=["risk_parity", "hrp", "erc", "robust_mv"],
        expected_effect="Portfolio weights distribution",
        economic_intuition="Different schemes allocate differently based on risk",
    ),
    # === Risk Settings ===
    SettingDef(
        key="risk_target",
        label="Risk Target",
        category="Risk",
        baseline=0.10,
        test_values=[0.05, 0.15, 0.20],
        expected_effect="Portfolio volatility scaling",
        economic_intuition="Higher target → higher scaled volatility",
    ),
    SettingDef(
        key="vol_floor",
        label="Volatility Floor",
        category="Risk",
        baseline=0.015,
        test_values=[0.01, 0.03, 0.05],
        expected_effect="Minimum volatility for scaling",
        economic_intuition="Higher floor → less extreme scaling for low-vol assets",
    ),
    SettingDef(
        key="max_weight",
        label="Max Weight",
        category="Constraints",
        baseline=0.20,
        test_values=[0.10, 0.15, 0.30],
        expected_effect="Maximum position size",
        economic_intuition="Lower cap → more diversified weights",
    ),
    SettingDef(
        key="min_weight",
        label="Min Weight",
        category="Constraints",
        baseline=0.05,
        test_values=[0.02, 0.08, 0.10],
        expected_effect="Minimum position size",
        economic_intuition="Higher floor → fewer very small positions",
    ),
    SettingDef(
        key="leverage_cap",
        label="Leverage Cap",
        category="Risk",
        baseline=2.0,
        test_values=[1.0, 1.5, 3.0],
        expected_effect="Gross exposure limit",
        economic_intuition="Lower cap → constrained gross exposure",
    ),
    # === Entry/Exit Thresholds ===
    SettingDef(
        key="z_entry_soft",
        label="Entry Threshold (Z)",
        category="Entry/Exit",
        baseline=1.0,
        test_values=[0.5, 1.5, 2.0],
        expected_effect="Frequency of fund entries",
        economic_intuition="Higher threshold → fewer entries (harder to qualify)",
    ),
    SettingDef(
        key="z_exit_soft",
        label="Exit Threshold (Z)",
        category="Entry/Exit",
        baseline=-1.0,
        test_values=[-0.5, -1.5, -2.0],
        expected_effect="Frequency of fund exits",
        economic_intuition="Higher (less negative) → more exits (easier to fail)",
    ),
    SettingDef(
        key="soft_strikes",
        label="Exit Strikes Required",
        category="Entry/Exit",
        baseline=2,
        test_values=[1, 3, 5],
        expected_effect="Exit frequency",
        economic_intuition="More strikes → slower exits (more chances)",
    ),
    SettingDef(
        key="entry_soft_strikes",
        label="Entry Strikes Required",
        category="Entry/Exit",
        baseline=1,
        test_values=[2, 3, 4],
        expected_effect="Entry frequency",
        economic_intuition="More strikes → slower entries (must prove consistency)",
    ),
    # === Multi-Period Settings ===
    SettingDef(
        key="lookback_periods",
        label="Lookback Periods",
        category="Multi-Period",
        baseline=3,
        test_values=[2, 5, 7],
        expected_effect="In-sample window length",
        economic_intuition="Longer lookback → more historical data for ranking",
    ),
    SettingDef(
        key="evaluation_periods",
        label="Evaluation Periods",
        category="Multi-Period",
        baseline=1,
        test_values=[2, 3],
        expected_effect="Out-of-sample window length",
        economic_intuition="Longer evaluation → longer hold before rebalance",
    ),
    SettingDef(
        key="multi_period_frequency",
        label="Period Frequency",
        category="Multi-Period",
        baseline="A",
        test_values=["Q", "M"],
        expected_effect="Number of periods generated",
        economic_intuition="Higher frequency → more periods",
    ),
    SettingDef(
        key="mp_max_funds",
        label="Max Funds per Period",
        category="Multi-Period",
        baseline=25,
        test_values=[10, 15, 30],
        expected_effect="Maximum portfolio size",
        economic_intuition="Lower cap → fewer funds allowed",
    ),
    SettingDef(
        key="mp_min_funds",
        label="Min Funds per Period",
        category="Multi-Period",
        baseline=5,
        test_values=[8, 12, 15],
        expected_effect="Minimum portfolio size",
        economic_intuition="Higher floor → must hold more funds",
    ),
    # === Holding Rules ===
    SettingDef(
        key="cooldown_periods",
        label="Cooldown Periods",
        category="Holding Rules",
        baseline=1,
        test_values=[0, 3, 5],
        expected_effect="Re-entry frequency for dropped funds",
        economic_intuition="Longer cooldown → less fund churning",
    ),
    SettingDef(
        key="min_tenure_periods",
        label="Min Tenure Periods",
        category="Holding Rules",
        baseline=3,
        test_values=[1, 5, 8],
        expected_effect="Average holding duration",
        economic_intuition="Longer tenure → forced to hold longer",
    ),
    SettingDef(
        key="min_weight_strikes",
        label="Underweight Strikes",
        category="Holding Rules",
        baseline=2,
        test_values=[1, 4, 6],
        expected_effect="Underweight exit frequency",
        economic_intuition="More strikes → slower to exit underweights",
    ),
    # === Costs ===
    SettingDef(
        key="transaction_cost_bps",
        label="Transaction Costs (bps)",
        category="Costs",
        baseline=0,
        test_values=[10, 30, 50],
        expected_effect="Total transaction costs",
        economic_intuition="Higher costs → lower net returns",
    ),
    SettingDef(
        key="slippage_bps",
        label="Slippage (bps)",
        category="Costs",
        baseline=0,
        test_values=[5, 15, 25],
        expected_effect="Total costs from market impact",
        economic_intuition="Higher slippage → lower net returns",
    ),
    SettingDef(
        key="max_turnover",
        label="Max Turnover",
        category="Costs",
        baseline=1.0,
        test_values=[0.3, 0.5, 0.7],
        expected_effect="Actual portfolio turnover",
        economic_intuition="Lower cap → more constrained rebalancing",
    ),
    # === Metric Weights ===
    SettingDef(
        key="metric_weights",
        label="Metric Weights",
        category="Scoring",
        baseline={"sharpe": 1.0, "return_ann": 1.0, "drawdown": 0.5},
        test_values=[
            {"sharpe": 3.0, "return_ann": 0.0, "drawdown": 0.0},
            {"sharpe": 0.0, "return_ann": 3.0, "drawdown": 0.0},
            {"sharpe": 0.0, "return_ann": 0.0, "drawdown": 3.0},
        ],
        expected_effect="Which funds get selected",
        economic_intuition="Different weights prioritize different fund characteristics",
    ),
    # === Robustness ===
    SettingDef(
        key="shrinkage_enabled",
        label="Shrinkage Enabled",
        category="Robustness",
        baseline=True,
        test_values=[False],
        expected_effect="Weight stability and estimation",
        economic_intuition="Shrinkage → more stable covariance estimates",
    ),
    SettingDef(
        key="shrinkage_method",
        label="Shrinkage Method",
        category="Robustness",
        baseline="ledoit_wolf",
        test_values=["oas"],
        expected_effect="Covariance estimation approach",
        economic_intuition="Different methods → different weight allocations",
    ),
    # === Data/Preprocessing ===
    SettingDef(
        key="missing_policy",
        label="Missing Data Policy",
        category="Data",
        baseline="ffill",
        test_values=["drop", "zero"],
        expected_effect="How missing data is handled",
        economic_intuition="Different policies → different data availability",
    ),
    SettingDef(
        key="winsorize_enabled",
        label="Winsorize Enabled",
        category="Data",
        baseline=True,
        test_values=[False],
        expected_effect="Treatment of extreme returns",
        economic_intuition="Winsorization → less impact from outliers",
    ),
    SettingDef(
        key="warmup_periods",
        label="Warmup Periods",
        category="Data",
        baseline=0,
        test_values=[6, 12, 24],
        expected_effect="Effective analysis start date",
        economic_intuition="Warmup → ignore early unstable estimates",
    ),
    # === Signals ===
    SettingDef(
        key="trend_window",
        label="Trend Window",
        category="Signals",
        baseline=63,
        test_values=[21, 42, 126],
        expected_effect="Signal smoothness/responsiveness",
        economic_intuition="Shorter window → more responsive signals",
    ),
    SettingDef(
        key="trend_zscore",
        label="Z-Score Signals",
        category="Signals",
        baseline=False,
        test_values=[True],
        expected_effect="Signal distribution",
        economic_intuition="Z-score → normalized cross-sectional signals",
    ),
    # === Reproducibility ===
    SettingDef(
        key="random_seed",
        label="Random Seed",
        category="Reproducibility",
        baseline=42,
        test_values=[123, 456, 789],
        expected_effect="Random selection outcomes",
        economic_intuition="Different seeds → different random samples",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_baseline_state() -> dict[str, Any]:
    """Return baseline model state for testing."""
    return {
        "preset": "Test",
        "lookback_periods": 3,
        "min_history_periods": 3,
        "evaluation_periods": 1,
        "selection_count": 10,
        "weighting_scheme": "equal",
        "metric_weights": {"sharpe": 1.0, "return_ann": 1.0, "drawdown": 0.5},
        "risk_target": 0.10,
        "date_mode": "relative",
        "rf_override_enabled": False,
        "rf_rate_annual": 0.0,
        "vol_floor": 0.015,
        "warmup_periods": 0,
        "max_weight": 0.20,
        "min_weight": 0.05,
        "cooldown_periods": 1,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        "min_tenure_periods": 3,
        "max_changes_per_period": 0,
        "max_active_positions": 0,
        "trend_window": 63,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": False,
        "trend_vol_adjust": False,
        "trend_vol_target": None,
        "regime_enabled": False,
        "regime_proxy": "SPX",
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "leverage_cap": 2.0,
        "random_seed": 42,
        "condition_threshold": 1.0e12,
        "safe_mode": "hrp",
        "long_only": True,
        "missing_policy": "ffill",
        "winsorize_enabled": True,
        "winsorize_lower": 1.0,
        "winsorize_upper": 99.0,
        "z_entry_soft": 1.0,
        "z_exit_soft": -1.0,
        "soft_strikes": 2,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 2,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "inclusion_approach": "threshold",
        "buy_hold_initial": "top_n",
        "slippage_bps": 0,
        "bottom_k": 0,
        "rank_pct": 0.10,
        "mp_min_funds": 5,
        "mp_max_funds": 25,
        "z_entry_hard": None,
        "z_exit_hard": None,
    }


def extract_key_metrics(result: Any) -> dict[str, Any]:
    """Extract key metrics from analysis result (RunResult object) for comparison."""
    metrics = {}

    # Handle RunResult object attributes
    # Number of funds selected (from weights)
    if hasattr(result, "weights") and result.weights is not None:
        pos_weights = result.weights[result.weights > 0]
        metrics["num_funds"] = len(pos_weights)
        metrics["selected_funds"] = sorted(pos_weights.index.tolist())[:5]
        metrics["max_weight"] = float(result.weights.max())
        metrics["min_weight"] = (
            float(pos_weights.min()) if len(pos_weights) > 0 else 0.0
        )
        metrics["weight_std"] = float(result.weights.std())

    # Multi-period results
    if hasattr(result, "period_results") and result.period_results:
        periods = result.period_results
        metrics["num_periods"] = len(periods)
        fund_counts = []
        for p in periods:
            if "weights" in p:
                w = p["weights"]
                fund_counts.append(sum(1 for v in w.values() if v > 0))
        if fund_counts:
            metrics["avg_funds_per_period"] = np.mean(fund_counts)
            metrics["min_funds_period"] = min(fund_counts)
            metrics["max_funds_period"] = max(fund_counts)

    # Period count attribute
    if hasattr(result, "period_count"):
        metrics["period_count"] = result.period_count

    # Summary metrics from metrics DataFrame
    if hasattr(result, "metrics") and result.metrics is not None:
        df = result.metrics
        if "CAGR" in df.columns:
            metrics["portfolio_CAGR"] = float(df["CAGR"].iloc[0])
        if "Volatility" in df.columns:
            metrics["portfolio_Volatility"] = float(df["Volatility"].iloc[0])
        if "Sharpe" in df.columns:
            metrics["portfolio_Sharpe"] = float(df["Sharpe"].iloc[0])
        if "Sortino" in df.columns:
            metrics["portfolio_Sortino"] = float(df["Sortino"].iloc[0])
        if "MaxDrawdown" in df.columns:
            metrics["portfolio_MaxDrawdown"] = float(df["MaxDrawdown"].iloc[0])

    # Costs from result
    if hasattr(result, "costs") and result.costs:
        for key in ["total", "transaction", "slippage"]:
            if key in result.costs:
                metrics[f"cost_{key}"] = result.costs[key]

    # Turnover
    if hasattr(result, "turnover") and result.turnover is not None:
        metrics["avg_turnover"] = float(result.turnover.mean())

    # Create a hash of the full result for change detection
    try:
        hashable = json.dumps(
            {k: str(v)[:50] for k, v in metrics.items()},
            sort_keys=True,
        )
        metrics["_result_hash"] = hashlib.md5(hashable.encode()).hexdigest()[:8]
    except Exception:
        metrics["_result_hash"] = "error"

    return metrics


def run_test_analysis(
    returns: pd.DataFrame,
    model_state: dict[str, Any],
) -> dict[str, Any]:
    """Run analysis and return results with error handling."""
    try:
        payload = analysis_runner.AnalysisPayload(
            returns=returns,
            model_state=model_state,
            benchmark=None,
        )
        result = analysis_runner._execute_analysis(payload)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.2f}"
    if isinstance(value, dict):
        return json.dumps(value, indent=1)
    if isinstance(value, list):
        if len(value) > 3:
            return f"{value[:3]}... ({len(value)} items)"
        return str(value)
    return str(value)


# =============================================================================
# Main Page
# =============================================================================


def render_validation_page() -> None:
    """Render the Settings Validation page."""

    st.set_page_config(
        page_title="Settings Validation",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Settings Validation")
    st.markdown(
        """
    This page systematically tests each UI setting to verify it's properly
    connected to the analysis pipeline. Select a setting, choose test values,
    and compare the results.
    """
    )

    # Check if we have data loaded
    app_data = st.session_state.get("app_data")
    if app_data is None or app_data.get("returns") is None:
        st.warning("⚠️ Please load data on the Data page first.")
        st.stop()

    returns = app_data["returns"]

    # Organize settings by category
    categories = {}
    for setting in TESTABLE_SETTINGS:
        if setting.category not in categories:
            categories[setting.category] = []
        categories[setting.category].append(setting)

    # Sidebar for settings selection
    with st.sidebar:
        st.header("Test Configuration")

        category = st.selectbox(
            "Category",
            options=list(categories.keys()),
            index=0,
        )

        settings_in_cat = categories[category]
        setting_labels = [s.label for s in settings_in_cat]

        selected_label = st.selectbox(
            "Setting to Test",
            options=setting_labels,
            index=0,
        )

        selected_setting = next(s for s in settings_in_cat if s.label == selected_label)

        st.markdown("---")
        st.markdown(f"**Key:** `{selected_setting.key}`")
        st.markdown(f"**Baseline:** `{selected_setting.baseline}`")
        st.markdown(f"**Expected Effect:** {selected_setting.expected_effect}")
        st.markdown(f"**Economic Intuition:** {selected_setting.economic_intuition}")

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline Configuration")
        baseline_value = selected_setting.baseline
        st.code(f"{selected_setting.key} = {format_value(baseline_value)}")

    with col2:
        st.subheader("Test Configuration")
        test_value = st.selectbox(
            "Select test value",
            options=selected_setting.test_values,
            format_func=format_value,
        )
        st.code(f"{selected_setting.key} = {format_value(test_value)}")

    # Run test button
    if st.button("🧪 Run Comparison Test", type="primary"):
        with st.spinner("Running baseline analysis..."):
            baseline_state = get_baseline_state()
            baseline_state[selected_setting.key] = baseline_value

            # Handle special cases
            if selected_setting.key == "inclusion_approach":
                if baseline_value == "random":
                    baseline_state["random_seed"] = 42

            baseline_result = run_test_analysis(returns, baseline_state)

        with st.spinner("Running test analysis..."):
            test_state = get_baseline_state()
            test_state[selected_setting.key] = test_value

            # Handle special cases
            if selected_setting.key == "inclusion_approach":
                if test_value == "random":
                    test_state["random_seed"] = 42
                elif test_value == "top_pct":
                    test_state["rank_pct"] = 0.20

            test_result = run_test_analysis(returns, test_state)

        # Display results
        st.markdown("---")
        st.subheader("Results Comparison")

        if baseline_result["status"] == "error":
            st.error(f"Baseline run failed: {baseline_result['error']}")
        elif test_result["status"] == "error":
            st.error(f"Test run failed: {test_result['error']}")
        else:
            baseline_metrics = extract_key_metrics(baseline_result["result"])
            test_metrics = extract_key_metrics(test_result["result"])

            # Check if anything changed
            baseline_hash = baseline_metrics.get("_result_hash", "")
            test_hash = test_metrics.get("_result_hash", "")

            if baseline_hash == test_hash:
                st.error(
                    f"""
                ❌ **SETTING NOT WIRED**: The setting `{selected_setting.key}` had 
                **no effect** on the analysis results!
                
                Baseline value: `{format_value(baseline_value)}`
                Test value: `{format_value(test_value)}`
                
                Both runs produced identical results. This setting may not be
                connected to the analysis pipeline.
                """
                )
            else:
                st.success(
                    f"""
                ✅ **SETTING IS WIRED**: The setting `{selected_setting.key}` 
                **did change** the analysis results.
                """
                )

            # Show detailed comparison
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.markdown("**Metric**")
            with col2:
                st.markdown("**Baseline**")
            with col3:
                st.markdown("**Test**")

            all_keys = set(baseline_metrics.keys()) | set(test_metrics.keys())
            all_keys = [k for k in sorted(all_keys) if not k.startswith("_")]

            for key in all_keys:
                baseline_val = baseline_metrics.get(key, "N/A")
                test_val = test_metrics.get(key, "N/A")

                changed = baseline_val != test_val

                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    st.markdown(f"`{key}`")
                with col2:
                    st.markdown(format_value(baseline_val))
                with col3:
                    if changed:
                        st.markdown(f"**{format_value(test_val)}** 🔄")
                    else:
                        st.markdown(format_value(test_val))

            # Store results for download
            st.session_state["last_validation_test"] = {
                "setting": selected_setting.key,
                "baseline_value": baseline_value,
                "test_value": test_value,
                "baseline_metrics": baseline_metrics,
                "test_metrics": test_metrics,
                "changed": baseline_hash != test_hash,
                "timestamp": datetime.now().isoformat(),
            }

    # Batch test section
    st.markdown("---")
    st.subheader("🔬 Batch Validation")

    with st.expander("Run All Tests in Category"):
        if st.button(f"Test All {category} Settings"):
            results = []
            progress = st.progress(0)
            status_text = st.empty()

            for i, setting in enumerate(settings_in_cat):
                status_text.text(f"Testing: {setting.label}...")

                # Run baseline
                baseline_state = get_baseline_state()
                baseline_state[setting.key] = setting.baseline
                baseline_result = run_test_analysis(returns, baseline_state)

                # Run first test value
                test_state = get_baseline_state()
                test_state[setting.key] = setting.test_values[0]
                test_result = run_test_analysis(returns, test_state)

                if (
                    baseline_result["status"] == "success"
                    and test_result["status"] == "success"
                ):
                    baseline_metrics = extract_key_metrics(baseline_result["result"])
                    test_metrics = extract_key_metrics(test_result["result"])

                    changed = baseline_metrics.get("_result_hash") != test_metrics.get(
                        "_result_hash"
                    )
                    status = "✅ WIRED" if changed else "❌ NOT WIRED"
                else:
                    status = "⚠️ ERROR"
                    changed = None

                results.append(
                    {
                        "Setting": setting.label,
                        "Key": setting.key,
                        "Baseline": str(setting.baseline),
                        "Test": str(setting.test_values[0]),
                        "Status": status,
                        "Expected Effect": setting.expected_effect,
                    }
                )

                progress.progress((i + 1) / len(settings_in_cat))

            status_text.text("Complete!")

            # Show results table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Summary
            wired = sum(
                1
                for r in results
                if "WIRED" in r["Status"] and "NOT" not in r["Status"]
            )
            not_wired = sum(1 for r in results if "NOT WIRED" in r["Status"])
            errors = sum(1 for r in results if "ERROR" in r["Status"])

            st.markdown(
                f"""
            **Summary for {category}:**
            - ✅ Wired: {wired}
            - ❌ Not Wired: {not_wired}
            - ⚠️ Errors: {errors}
            """
            )

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"settings_validation_{category}_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )

    # Full validation section
    with st.expander("Run Full Validation (All Settings)"):
        st.warning("⚠️ This will test all settings and may take several minutes.")

        if st.button("🚀 Run Full Validation"):
            all_results = []
            progress = st.progress(0)
            status_text = st.empty()

            total = len(TESTABLE_SETTINGS)

            for i, setting in enumerate(TESTABLE_SETTINGS):
                status_text.text(f"[{i+1}/{total}] Testing: {setting.label}...")

                # Run baseline
                baseline_state = get_baseline_state()
                baseline_state[setting.key] = setting.baseline
                baseline_result = run_test_analysis(returns, baseline_state)

                # Run first test value
                test_state = get_baseline_state()
                test_state[setting.key] = setting.test_values[0]

                # Handle special cases
                if setting.key == "inclusion_approach":
                    if setting.test_values[0] == "random":
                        test_state["random_seed"] = 42
                    elif setting.test_values[0] == "top_pct":
                        test_state["rank_pct"] = 0.20

                test_result = run_test_analysis(returns, test_state)

                if (
                    baseline_result["status"] == "success"
                    and test_result["status"] == "success"
                ):
                    baseline_metrics = extract_key_metrics(baseline_result["result"])
                    test_metrics = extract_key_metrics(test_result["result"])

                    changed = baseline_metrics.get("_result_hash") != test_metrics.get(
                        "_result_hash"
                    )
                    status = "✅ WIRED" if changed else "❌ NOT WIRED"
                else:
                    status = "⚠️ ERROR"
                    # Error details available in baseline_result/test_result
                    changed = None

                all_results.append(
                    {
                        "Category": setting.category,
                        "Setting": setting.label,
                        "Key": setting.key,
                        "Baseline": str(setting.baseline)[:30],
                        "Test": str(setting.test_values[0])[:30],
                        "Status": status,
                        "Expected Effect": setting.expected_effect,
                    }
                )

                progress.progress((i + 1) / total)

            status_text.text("Complete!")

            # Show results
            results_df = pd.DataFrame(all_results)
            st.dataframe(results_df, use_container_width=True)

            # Summary by category
            st.subheader("Summary by Category")
            for cat in categories.keys():
                cat_results = [r for r in all_results if r["Category"] == cat]
                wired = sum(1 for r in cat_results if "✅" in r["Status"])
                not_wired = sum(1 for r in cat_results if "❌" in r["Status"])
                errors = sum(1 for r in cat_results if "⚠️" in r["Status"])

                if not_wired > 0:
                    st.error(f"**{cat}**: {wired}✅ / {not_wired}❌ / {errors}⚠️")
                elif errors > 0:
                    st.warning(f"**{cat}**: {wired}✅ / {not_wired}❌ / {errors}⚠️")
                else:
                    st.success(f"**{cat}**: {wired}✅ / {not_wired}❌ / {errors}⚠️")

            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results CSV",
                data=csv,
                file_name=f"full_settings_validation_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    render_validation_page()
