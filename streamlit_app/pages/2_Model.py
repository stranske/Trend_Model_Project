"""Model configuration page for the Streamlit application."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner

# Extended metric fields for ranking
METRIC_FIELDS = [
    ("Sharpe Ratio", "sharpe"),
    ("Annual Return", "return_ann"),
    ("Sortino Ratio", "sortino"),
    ("Info Ratio", "info_ratio"),
    ("Max Drawdown", "drawdown"),
    ("Volatility", "vol"),
]

# Available weighting schemes from the plugin registry
WEIGHTING_SCHEMES = [
    ("Equal Weight (1/N)", "equal"),
    ("Risk Parity (inverse vol)", "risk_parity"),
    ("Hierarchical Risk Parity", "hrp"),
    ("Equal Risk Contribution", "erc"),
    ("Robust Mean-Variance", "robust_mv"),
    ("Robust Risk Parity", "robust_risk_parity"),
]

# Preset configurations with default parameter values
PRESET_CONFIGS = {
    "Baseline": {
        "lookback_months": 36,
        "min_history_months": 36,
        "evaluation_months": 12,
        "selection_count": 10,
        "weighting_scheme": "equal",
        "metric_weights": {
            "sharpe": 1.0,
            "return_ann": 1.0,
            "sortino": 0.0,
            "info_ratio": 0.0,
            "drawdown": 0.5,
            "vol": 0.0,
        },
        "risk_target": 0.10,
        # Date mode: "relative" (use lookback/eval windows) or "explicit" (user-specified dates)
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings
        "rf_rate_annual": 0.0,
        "vol_floor": 0.015,
        "warmup_periods": 0,
        # Volatility adjustment details (Phase 10)
        "vol_adjust_enabled": True,
        "vol_window_length": 63,
        "vol_window_decay": "ewma",
        "vol_ewma_lambda": 0.94,
        # Advanced settings
        "max_weight": 0.20,
        "cooldown_months": 3,
        "min_track_months": 24,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules (Phase 3)
        "min_tenure_periods": 3,
        "max_changes_per_period": 0,  # 0 = unlimited
        "max_active_positions": 0,  # 0 = unlimited (uses selection_count)
        # Trend signal parameters (Phase 4)
        "trend_window": 63,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": False,
        "trend_vol_adjust": False,
        "trend_vol_target": None,
        # Regime analysis (Phase 6)
        "regime_enabled": False,
        "regime_proxy": "SPX",
        # Robustness & Expert settings (Phase 7)
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "leverage_cap": 2.0,
        "random_seed": 42,
        # Robustness fallbacks (Phase 14)
        "condition_threshold": 1.0e12,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Data/Preprocessing (Phase 16)
        "missing_policy": "ffill",
        "winsorize_enabled": True,
        "winsorize_lower": 1.0,
        "winsorize_upper": 99.0,
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": 1.0,
        "z_exit_soft": -1.0,
        "soft_strikes": 2,
        "entry_soft_strikes": 1,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": False,
        "multi_period_frequency": "A",
        "in_sample_years": 3,
        "out_sample_years": 1,
        "inclusion_approach": "top_n",
        "rank_transform": "none",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        "rank_threshold": 1.5,
        # Multi-period bounds (Phase 12)
        "mp_min_funds": 10,
        "mp_max_funds": 25,
        # Hard thresholds (Phase 13)
        "z_entry_hard": None,
        "z_exit_hard": None,
    },
    "Conservative": {
        "lookback_months": 48,
        "min_history_months": 48,
        "evaluation_months": 12,
        "selection_count": 8,
        "weighting_scheme": "risk_parity",
        "metric_weights": {
            "sharpe": 1.0,
            "return_ann": 0.5,
            "sortino": 1.0,
            "info_ratio": 0.0,
            "drawdown": 1.5,
            "vol": 1.0,
        },
        "risk_target": 0.08,
        # Date mode
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings - lower floor for more conservative scaling
        "rf_rate_annual": 0.0,
        "vol_floor": 0.02,
        "warmup_periods": 6,
        # Advanced settings - more restrictive
        "max_weight": 0.15,
        "cooldown_months": 6,
        "min_track_months": 36,
        "rebalance_freq": "Q",
        "max_turnover": 0.50,
        "transaction_cost_bps": 10,
        # Fund holding rules - conservative: higher tenure, limited changes
        "min_tenure_periods": 6,
        "max_changes_per_period": 2,
        "max_active_positions": 10,
        # Trend signal parameters - longer window for stability
        "trend_window": 126,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": True,
        "trend_vol_adjust": False,
        "trend_vol_target": None,
        # Regime analysis - enabled for defensive positioning
        "regime_enabled": True,
        "regime_proxy": "SPX",
        # Robustness & Expert settings - more conservative
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "leverage_cap": 1.5,
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - conservative: stricter threshold
        "condition_threshold": 1.0e10,
        "safe_mode": "risk_parity",
        # Constraints (Phase 15)
        "long_only": True,
        # Data/Preprocessing (Phase 16) - conservative: stricter cleaning
        "missing_policy": "drop",
        "winsorize_enabled": True,
        "winsorize_lower": 2.0,
        "winsorize_upper": 98.0,
        # Entry/Exit thresholds - conservative: stricter entry, lenient exit
        "z_entry_soft": 1.5,
        "z_exit_soft": -1.0,
        "soft_strikes": 3,
        "entry_soft_strikes": 2,
        "sticky_add_periods": 2,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - conservative: longer periods
        "multi_period_enabled": False,
        "multi_period_frequency": "A",
        "in_sample_years": 5,
        "out_sample_years": 1,
        "inclusion_approach": "top_n",
        "rank_transform": "zscore",
        "slippage_bps": 5,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        "rank_threshold": 2.0,  # stricter for conservative
        # Multi-period bounds (Phase 12) - conservative: narrower range
        "mp_min_funds": 8,
        "mp_max_funds": 15,
        # Hard thresholds (Phase 13) - conservative: enabled, stricter
        "z_entry_hard": 2.5,
        "z_exit_hard": -2.5,
    },
    "Aggressive": {
        "lookback_months": 24,
        "min_history_months": 24,
        "evaluation_months": 6,
        "selection_count": 15,
        "weighting_scheme": "hrp",
        "metric_weights": {
            "sharpe": 0.5,
            "return_ann": 2.0,
            "sortino": 0.5,
            "info_ratio": 0.0,
            "drawdown": 0.0,
            "vol": 0.0,
        },
        "risk_target": 0.15,
        # Date mode
        "date_mode": "relative",
        "start_date": None,
        "end_date": None,
        # Risk settings - lower floor, no warmup for faster response
        "rf_rate_annual": 0.0,
        "vol_floor": 0.01,
        "warmup_periods": 0,
        # Advanced settings - less restrictive
        "max_weight": 0.25,
        "cooldown_months": 1,
        "min_track_months": 12,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules - aggressive: minimal constraints
        "min_tenure_periods": 1,
        "max_changes_per_period": 0,  # unlimited
        "max_active_positions": 0,  # unlimited
        # Trend signal parameters - shorter window for responsiveness
        "trend_window": 42,
        "trend_lag": 1,
        "trend_min_periods": None,
        "trend_zscore": False,
        "trend_vol_adjust": True,
        "trend_vol_target": 0.10,
        # Regime analysis - disabled for pure momentum
        "regime_enabled": False,
        "regime_proxy": "SPX",
        # Robustness & Expert settings - more flexibility
        "shrinkage_enabled": True,
        "shrinkage_method": "ledoit_wolf",
        "leverage_cap": 3.0,
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - aggressive: higher tolerance
        "condition_threshold": 1.0e14,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Data/Preprocessing (Phase 16) - aggressive: less cleaning
        "missing_policy": "ffill",
        "winsorize_enabled": False,
        "winsorize_lower": 1.0,
        "winsorize_upper": 99.0,
        # Entry/Exit thresholds - aggressive: lenient entry, quick exit
        "z_entry_soft": 0.5,
        "z_exit_soft": -0.5,
        "soft_strikes": 1,
        "entry_soft_strikes": 1,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - aggressive: shorter periods
        "multi_period_enabled": False,
        "multi_period_frequency": "Q",
        "in_sample_years": 2,
        "out_sample_years": 1,
        "inclusion_approach": "top_n",
        "rank_transform": "none",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.15,  # more aggressive percentage
        "rank_threshold": 1.0,  # lower threshold for aggressive
        # Multi-period bounds (Phase 12) - aggressive: wider range
        "mp_min_funds": 15,
        "mp_max_funds": 40,
        # Hard thresholds (Phase 13) - aggressive: disabled
        "z_entry_hard": None,
        "z_exit_hard": None,
    },
    "Custom": None,  # Custom means keep current values
}

# Common index/benchmark column names
BENCHMARK_COLUMNS = ["SPX", "TSX", "MSCI", "ACWI", "EAFE", "EM", "AGG", "BND"]

# Help text for configuration parameters (brief tooltips)
HELP_TEXT = {
    "preset": "Pre-configured settings optimized for different investment styles. Changing preset auto-populates all parameters.",
    "lookback": "Months of history used to calculate fund metrics (Sharpe, returns, etc.) for ranking.",
    "min_history": "Minimum months of data required for a fund to be considered for selection.",
    "evaluation": "Out-of-sample period (months) to measure portfolio performance after selection.",
    "selection": "Number of top-ranked funds to include in the portfolio.",
    "weighting": "How to allocate capital across selected funds. See Help page for details.",
    "sharpe_weight": "Importance of risk-adjusted returns in fund ranking.",
    "return_weight": "Importance of absolute returns in fund ranking.",
    "sortino_weight": "Importance of downside risk-adjusted returns in fund ranking.",
    "info_ratio_weight": "Importance of benchmark-relative risk-adjusted returns.",
    "drawdown_weight": "Importance of limiting drawdowns in fund ranking.",
    "vol_weight": "Importance of low volatility in fund ranking (lower vol = higher rank).",
    "risk_target": "Target portfolio volatility. Weights are scaled to achieve this level.",
    "info_ratio_benchmark": "Benchmark for calculating Information Ratio. Select an index or fund column.",
    # Date range settings
    "date_mode": "Choose whether to use relative lookback windows or explicit start/end dates.",
    "start_date": "Simulation start date. Data before this date will be excluded.",
    "end_date": "Simulation end date. Data after this date will be excluded.",
    # Risk settings
    "rf_rate": "Annual risk-free rate used for Sharpe/Sortino calculations. Default: 0%.",
    "vol_floor": "Minimum volatility floor for scaling. Prevents extreme weights on low-vol assets.",
    "warmup_periods": "Initial periods with zero portfolio weight (warm-up for signals).",
    # Phase 10: Volatility adjustment details
    "vol_adjust_enabled": "Enable volatility adjustment to scale returns to target vol.",
    "vol_window_length": "Rolling window for volatility estimation (periods). ~63 = 3 months.",
    "vol_window_decay": "EWMA weights recent data more; Simple uses equal weights.",
    "vol_ewma_lambda": "EWMA decay factor. Higher = longer memory. 0.94 is RiskMetrics standard.",
    # Advanced settings
    "max_weight": "Maximum allocation to any single fund. Prevents concentration risk.",
    "cooldown_months": "After a fund is removed, it cannot be re-added for this many months.",
    "min_track_months": "Minimum track record (months of data) required for a fund to be eligible.",
    "rebalance_freq": "How often to rebalance the portfolio weights.",
    "max_turnover": "Maximum portfolio turnover allowed per rebalance (1.0 = 100%).",
    "transaction_cost_bps": "Transaction cost in basis points (0.01% = 1 bp) applied per trade.",
    # Phase 3: Fund holding rules
    "min_tenure": "Minimum periods a fund must be held before it can be removed.",
    "max_changes": "Maximum number of fund additions/removals per rebalance. 0 = unlimited.",
    "max_active": "Maximum active positions in portfolio. 0 = use selection count.",
    # Phase 4: Trend signal parameters
    "trend_window": "Rolling window size for computing momentum signals (in periods).",
    "trend_lag": "Number of periods to lag the signal (minimum 1 for causality).",
    "trend_min_periods": "Minimum observations required in rolling window. Blank = use window.",
    "trend_zscore": "Cross-sectionally standardize signals at each time step.",
    "trend_vol_adjust": "Scale signals by volatility to normalize across assets.",
    "trend_vol_target": "Target volatility for vol-adjusted signals.",
    # Phase 6: Regime analysis
    "regime_enabled": "Enable regime detection to adjust behavior in risk-on/risk-off environments.",
    "regime_proxy": "Market index used to detect risk-on/risk-off regimes.",
    # Phase 7: Robustness & Expert settings
    "shrinkage_enabled": "Apply covariance matrix shrinkage to improve stability.",
    "shrinkage_method": "Shrinkage method: Ledoit-Wolf or Oracle Approximating Shrinkage.",
    "leverage_cap": "Maximum gross exposure (1.0 = no leverage).",
    "random_seed": "Random seed for reproducibility. Change for different random selections.",
    # Phase 5: Entry/Exit thresholds
    "z_entry_soft": "Z-score threshold for fund entry consideration. Higher = stricter entry.",
    "z_exit_soft": "Z-score threshold for fund exit consideration. Lower = stricter exit.",
    "soft_strikes": "Consecutive periods below exit threshold before removing a fund.",
    "entry_soft_strikes": "Consecutive periods above entry threshold before adding a fund.",
    "sticky_add_periods": "Periods a fund must rank highly before being added to portfolio.",
    "sticky_drop_periods": "Periods a fund must rank poorly before being removed from portfolio.",
    "ci_level": "Confidence interval level for entry gate (0 = disabled, 0.9 = 90% CI).",
    # Phase 8: Multi-period & Selection settings
    "multi_period_enabled": "Enable rolling multi-period walk-forward analysis.",
    "multi_period_frequency": "Period frequency: Monthly (M), Quarterly (Q), or Annual (A).",
    "in_sample_years": "Number of years for in-sample (training) window.",
    "out_sample_years": "Number of years for out-of-sample (testing) window.",
    "inclusion_approach": "How to select funds: Top N, Top Percentage, or Z-score Threshold.",
    "rank_transform": "Transform scores before ranking: None, Z-score, or Percentile Rank.",
    "slippage_bps": "Additional slippage cost in basis points (market impact).",
    "bottom_k": "Number of bottom-ranked funds to always exclude (0 = none).",
    # Phase 9: Selection approach details
    "rank_pct": "Percentage of funds to include (0.10 = top 10%). Used with Top Percentage approach.",
    "rank_threshold": "Z-score threshold for inclusion. Funds scoring above this are selected.",
    # Phase 12: Multi-period bounds
    "mp_min_funds": "Minimum number of funds to hold in multi-period analysis.",
    "mp_max_funds": "Maximum number of funds to hold in multi-period analysis.",
    # Phase 13: Hard entry/exit thresholds
    "z_entry_hard": "Hard entry: Z-score for immediate addition (bypasses strikes).",
    "z_exit_hard": "Hard exit: Z-score for immediate removal (bypasses strikes).",
    # Phase 14: Robustness fallbacks
    "condition_threshold": "Maximum acceptable condition number for covariance matrix.",
    "safe_mode": "Fallback weighting method when matrix is ill-conditioned.",
    # Phase 15: Constraints
    "long_only": "Enforce long-only positions (no short selling).",
    # Phase 16: Data/Preprocessing
    "missing_policy": "How to handle missing data: drop rows, forward-fill, or replace with zero.",
    "winsorize_enabled": "Clip extreme returns to reduce impact of outliers.",
    "winsorize_lower": "Lower percentile cutoff for winsorization (e.g., 1% = clip below 1st percentile).",
    "winsorize_upper": "Upper percentile cutoff for winsorization (e.g., 99% = clip above 99th percentile).",
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize metric weights to sum to 1.0."""
    total = sum(float(w or 0) for w in weights.values())
    if total <= 0:
        return weights
    return {k: round(v / total, 4) for k, v in weights.items()}


def _get_benchmark_columns(df) -> list[str]:
    """Identify potential benchmark columns in the dataset."""
    if df is None:
        return []
    all_cols = [str(c) for c in df.columns if str(c).upper() not in ("DATE", "INDEX")]
    # Prioritize known benchmark names, then include all columns as options
    benchmark_priority = []
    other_cols = []
    for col in all_cols:
        if col.upper() in [b.upper() for b in BENCHMARK_COLUMNS]:
            benchmark_priority.append(col)
        else:
            other_cols.append(col)
    return benchmark_priority + other_cols


def _validate_model(values: Mapping[str, Any], column_count: int) -> list[str]:
    errors: list[str] = []
    lookback = values.get("lookback_months", 36)
    min_history = values.get("min_history_months", lookback)
    if min_history > lookback:
        errors.append("Minimum history cannot exceed the lookback window.")
    selection = values.get("selection_count", 10)
    if column_count and selection > column_count:
        errors.append(
            f"Selection count ({selection}) cannot exceed available assets ({column_count})."
        )
    weights = values.get("metric_weights", {})
    if not any(float(w or 0) > 0 for w in weights.values()):
        errors.append("Provide at least one positive metric weight.")
    # Validate benchmark is set if info_ratio weight > 0
    if float(weights.get("info_ratio", 0)) > 0:
        if not values.get("info_ratio_benchmark"):
            errors.append("Select a benchmark for Information Ratio metric.")
    return errors


def _initial_model_state() -> dict[str, Any]:
    """Return default model state based on Baseline preset."""
    baseline = PRESET_CONFIGS["Baseline"]
    return {
        "preset": "Baseline",
        "lookback_months": baseline["lookback_months"],
        "min_history_months": baseline["min_history_months"],
        "evaluation_months": baseline["evaluation_months"],
        "selection_count": baseline["selection_count"],
        "weighting_scheme": baseline["weighting_scheme"],
        "metric_weights": baseline["metric_weights"].copy(),
        "risk_target": baseline["risk_target"],
        "info_ratio_benchmark": "",  # Empty until user selects
        # Date settings
        "date_mode": baseline["date_mode"],
        "start_date": baseline["start_date"],
        "end_date": baseline["end_date"],
        # Risk settings
        "rf_rate_annual": baseline["rf_rate_annual"],
        "vol_floor": baseline["vol_floor"],
        "warmup_periods": baseline["warmup_periods"],
        # Advanced settings
        "max_weight": baseline["max_weight"],
        "cooldown_months": baseline["cooldown_months"],
        "min_track_months": baseline["min_track_months"],
        "rebalance_freq": baseline["rebalance_freq"],
        "max_turnover": baseline["max_turnover"],
        "transaction_cost_bps": baseline["transaction_cost_bps"],
        # Fund holding rules (Phase 3)
        "min_tenure_periods": baseline["min_tenure_periods"],
        "max_changes_per_period": baseline["max_changes_per_period"],
        "max_active_positions": baseline["max_active_positions"],
        # Trend signal parameters (Phase 4)
        "trend_window": baseline["trend_window"],
        "trend_lag": baseline["trend_lag"],
        "trend_min_periods": baseline["trend_min_periods"],
        "trend_zscore": baseline["trend_zscore"],
        "trend_vol_adjust": baseline["trend_vol_adjust"],
        "trend_vol_target": baseline["trend_vol_target"],
        # Regime analysis (Phase 6)
        "regime_enabled": baseline["regime_enabled"],
        "regime_proxy": baseline["regime_proxy"],
        # Robustness & Expert settings (Phase 7)
        "shrinkage_enabled": baseline["shrinkage_enabled"],
        "shrinkage_method": baseline["shrinkage_method"],
        "leverage_cap": baseline["leverage_cap"],
        "random_seed": baseline["random_seed"],
        # Robustness fallbacks (Phase 14)
        "condition_threshold": baseline["condition_threshold"],
        "safe_mode": baseline["safe_mode"],
        # Constraints (Phase 15)
        "long_only": baseline["long_only"],
        # Data/Preprocessing (Phase 16)
        "missing_policy": baseline["missing_policy"],
        "winsorize_enabled": baseline["winsorize_enabled"],
        "winsorize_lower": baseline["winsorize_lower"],
        "winsorize_upper": baseline["winsorize_upper"],
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": baseline["z_entry_soft"],
        "z_exit_soft": baseline["z_exit_soft"],
        "soft_strikes": baseline["soft_strikes"],
        "entry_soft_strikes": baseline["entry_soft_strikes"],
        "sticky_add_periods": baseline["sticky_add_periods"],
        "sticky_drop_periods": baseline["sticky_drop_periods"],
        "ci_level": baseline["ci_level"],
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": baseline["multi_period_enabled"],
        "multi_period_frequency": baseline["multi_period_frequency"],
        "in_sample_years": baseline["in_sample_years"],
        "out_sample_years": baseline["out_sample_years"],
        "inclusion_approach": baseline["inclusion_approach"],
        "rank_transform": baseline["rank_transform"],
        "slippage_bps": baseline["slippage_bps"],
        "bottom_k": baseline["bottom_k"],
        # Selection approach details (Phase 9)
        "rank_pct": baseline["rank_pct"],
        "rank_threshold": baseline["rank_threshold"],
        # Multi-period bounds (Phase 12)
        "mp_min_funds": baseline["mp_min_funds"],
        "mp_max_funds": baseline["mp_max_funds"],
        # Hard thresholds (Phase 13)
        "z_entry_hard": baseline["z_entry_hard"],
        "z_exit_hard": baseline["z_exit_hard"],
    }


# Detailed descriptions for weighting schemes (shown in expander)
WEIGHTING_DESCRIPTIONS = {
    "equal": """
**Equal Weight (1/N)** allocates the same percentage to each selected fund.

- **Pros:** Simple, transparent, robust to estimation error
- **Cons:** Ignores risk characteristics; high-vol funds contribute more risk
- **Best for:** Most users; when you don't want to make assumptions about fund behavior
""",
    "risk_parity": """
**Risk Parity** allocates weights inversely proportional to each fund's volatility.
Higher-volatility funds receive lower weights, so each contributes roughly equal risk.

- **Pros:** Balances risk across assets; reduces concentration in volatile funds
- **Cons:** Ignores correlations; may over-allocate to low-vol assets
- **Best for:** Portfolios with assets of varying volatilities
""",
    "hrp": """
**Hierarchical Risk Parity (HRP)** uses machine learning clustering to build a
diversified allocation based on correlation structure.

- **Pros:** Accounts for correlations; more stable than mean-variance
- **Cons:** More complex; requires sufficient data for correlation estimation
- **Best for:** Complex portfolios with many correlated assets
""",
    "erc": """
**Equal Risk Contribution (ERC)** optimizes weights so each fund contributes
exactly the same marginal risk to the portfolio.

- **Pros:** Formal risk targeting; theoretically optimal risk allocation
- **Cons:** Requires optimization; sensitive to covariance estimation
- **Best for:** Formal risk management with specific risk targets
""",
    "robust_mv": """
**Robust Mean-Variance** uses shrinkage estimation to stabilize the covariance
matrix, reducing sensitivity to estimation error.

- **Pros:** More stable than classical MVO; resistant to extreme weights
- **Cons:** Still assumes you trust return forecasts
- **Best for:** When you have return forecasts but want protection from estimation error
""",
    "robust_risk_parity": """
**Robust Risk Parity** combines risk parity allocation with shrinkage estimation
for the covariance matrix.

- **Pros:** Benefits of risk parity with improved covariance estimation
- **Cons:** More complex; requires tuning shrinkage parameters
- **Best for:** Large portfolios with estimation uncertainty
""",
}


def render_model_page() -> None:
    app_state.initialize_session_state()
    st.title("Model Configuration")

    # Help link - use st.page_link for proper navigation
    st.markdown(
        "📖 Use the **Help** page in the sidebar for detailed explanations of all parameters."
    )

    df, meta = app_state.get_uploaded_data()
    if df is None:
        st.error("Load data on the Data page before configuring the model.")
        return

    # Display dataset summary with name
    st.markdown("---")
    st.subheader("📊 Dataset Summary")

    # Get dataset name from session state or meta
    dataset_name = st.session_state.get("uploaded_filename", "Demo Dataset")
    if meta and hasattr(meta, "source_name"):
        dataset_name = meta.source_name
    st.markdown(f"**Dataset:** {dataset_name}")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        # Count only fund columns (exclude benchmarks/indices)
        fund_cols = [
            c
            for c in df.columns
            if c.upper()
            not in ("DATE", "SPX", "TSX", "RF", "CASH", "TBILL", "TBILLS", "T-BILL")
        ]
        st.metric("Funds", len(fund_cols))
    with col_info2:
        st.metric("Time Periods", len(df))
    with col_info3:
        if hasattr(df.index, "min") and hasattr(df.index, "max"):
            start_date = df.index.min()
            st.metric(
                "Start Date",
                (
                    start_date.strftime("%Y-%m")
                    if hasattr(start_date, "strftime")
                    else str(start_date)[:7]
                ),
            )
    with col_info4:
        if hasattr(df.index, "max"):
            end_date = df.index.max()
            st.metric(
                "End Date",
                (
                    end_date.strftime("%Y-%m")
                    if hasattr(end_date, "strftime")
                    else str(end_date)[:7]
                ),
            )

    # Get data date boundaries for validation
    data_start = df.index.min() if hasattr(df.index, "min") else None
    data_end = df.index.max() if hasattr(df.index, "max") else None

    # Show fund names in an expander
    with st.expander("View fund names"):
        st.write(", ".join(fund_cols[:50]))
        if len(fund_cols) > 50:
            st.caption(f"...and {len(fund_cols) - 50} more")

    st.markdown("---")

    model_state = st.session_state.setdefault("model_state", _initial_model_state())

    # =============================================
    # SIMULATION PERIOD SETTINGS (outside form for immediate feedback)
    # =============================================
    st.subheader("📅 Simulation Period")
    st.caption("Define the time range for your simulation.")

    date_mode_options = ["relative", "explicit"]
    date_mode_labels = {
        "relative": "Relative (use lookback windows)",
        "explicit": "Explicit (specify start/end dates)",
    }
    current_date_mode = model_state.get("date_mode", "relative")

    date_mode = st.radio(
        "Date Mode",
        options=date_mode_options,
        format_func=lambda x: date_mode_labels.get(x, x),
        index=(
            date_mode_options.index(current_date_mode)
            if current_date_mode in date_mode_options
            else 0
        ),
        help=HELP_TEXT["date_mode"],
        horizontal=True,
        key="date_mode_radio",
    )

    # Update model state if date mode changed
    if date_mode != current_date_mode:
        st.session_state["model_state"]["date_mode"] = date_mode

    # Show date pickers when in explicit mode
    if date_mode == "explicit":
        date_col1, date_col2 = st.columns(2)

        # Convert data boundaries to date objects for the date picker
        if data_start is not None and hasattr(data_start, "date"):
            min_date = data_start.date()
        else:
            min_date = None

        if data_end is not None and hasattr(data_end, "date"):
            max_date = data_end.date()
        else:
            max_date = None

        # Get current values from model state
        current_start = model_state.get("start_date")
        current_end = model_state.get("end_date")

        # Convert to date objects if they're strings
        if isinstance(current_start, str) and current_start:
            try:
                import datetime

                current_start = datetime.datetime.strptime(
                    current_start[:7] + "-01", "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                current_start = min_date
        elif current_start is None:
            current_start = min_date

        if isinstance(current_end, str) and current_end:
            try:
                import datetime

                current_end = datetime.datetime.strptime(
                    current_end[:7] + "-01", "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                current_end = max_date
        elif current_end is None:
            current_end = max_date

        with date_col1:
            sim_start_date = st.date_input(
                "Simulation Start Date",
                value=current_start,
                min_value=min_date,
                max_value=max_date,
                help=HELP_TEXT["start_date"],
                key="sim_start_date",
            )
            # Update model state
            if sim_start_date:
                st.session_state["model_state"]["start_date"] = sim_start_date.strftime(
                    "%Y-%m-%d"
                )

        with date_col2:
            sim_end_date = st.date_input(
                "Simulation End Date",
                value=current_end,
                min_value=min_date,
                max_value=max_date,
                help=HELP_TEXT["end_date"],
                key="sim_end_date",
            )
            # Update model state
            if sim_end_date:
                st.session_state["model_state"]["end_date"] = sim_end_date.strftime(
                    "%Y-%m-%d"
                )

        # Validate date range
        if sim_start_date and sim_end_date and sim_start_date > sim_end_date:
            st.error("Start date must be before end date.")
        else:
            # Show selected period info
            if sim_start_date and sim_end_date:
                months_span = (sim_end_date.year - sim_start_date.year) * 12 + (
                    sim_end_date.month - sim_start_date.month
                )
                if months_span > 600:  # 50 years * 12 months
                    st.warning(
                        "Date range exceeds 50 years - please verify your selection."
                    )
                else:
                    st.info(
                        f"📊 Selected period: {sim_start_date.strftime('%Y-%m')} to {sim_end_date.strftime('%Y-%m')} ({months_span} months)"
                    )
    else:
        st.info(
            "📊 Using relative date mode: simulation dates will be computed from lookback and evaluation windows."
        )

    st.markdown("---")

    # Get benchmark column options for Info Ratio
    benchmark_options = _get_benchmark_columns(df)

    # Preset selection with auto-population (outside form for instant feedback)
    preset_options = ["Baseline", "Conservative", "Aggressive", "Custom"]
    current_preset = model_state.get("preset", "Baseline")
    try:
        preset_index = preset_options.index(current_preset)
    except ValueError:
        preset_index = 0

    # Preset selector (outside form for immediate updates)
    new_preset = st.selectbox(
        "📋 Preset Configuration",
        preset_options,
        index=preset_index,
        help=HELP_TEXT["preset"],
        key="preset_selector",
    )

    # Auto-populate when preset changes (except Custom)
    if new_preset != current_preset and new_preset != "Custom":
        preset_config = PRESET_CONFIGS.get(new_preset)
        if preset_config:
            st.session_state["model_state"] = {
                "preset": new_preset,
                "lookback_months": preset_config["lookback_months"],
                "min_history_months": preset_config["min_history_months"],
                "evaluation_months": preset_config["evaluation_months"],
                "selection_count": preset_config["selection_count"],
                "weighting_scheme": preset_config["weighting_scheme"],
                "metric_weights": preset_config["metric_weights"].copy(),
                "risk_target": preset_config["risk_target"],
                "info_ratio_benchmark": model_state.get("info_ratio_benchmark", ""),
                # Date settings
                "date_mode": preset_config["date_mode"],
                "start_date": preset_config["start_date"],
                "end_date": preset_config["end_date"],
                # Risk settings
                "rf_rate_annual": preset_config["rf_rate_annual"],
                "vol_floor": preset_config["vol_floor"],
                "warmup_periods": preset_config["warmup_periods"],
                # Advanced settings
                "max_weight": preset_config["max_weight"],
                "cooldown_months": preset_config["cooldown_months"],
                "min_track_months": preset_config["min_track_months"],
                "rebalance_freq": preset_config["rebalance_freq"],
                "max_turnover": preset_config["max_turnover"],
                "transaction_cost_bps": preset_config["transaction_cost_bps"],
                # Fund holding rules (Phase 3)
                "min_tenure_periods": preset_config["min_tenure_periods"],
                "max_changes_per_period": preset_config["max_changes_per_period"],
                "max_active_positions": preset_config["max_active_positions"],
                # Trend signal parameters (Phase 4)
                "trend_window": preset_config["trend_window"],
                "trend_lag": preset_config["trend_lag"],
                "trend_min_periods": preset_config["trend_min_periods"],
                "trend_zscore": preset_config["trend_zscore"],
                "trend_vol_adjust": preset_config["trend_vol_adjust"],
                "trend_vol_target": preset_config["trend_vol_target"],
                # Regime analysis (Phase 6)
                "regime_enabled": preset_config["regime_enabled"],
                "regime_proxy": preset_config["regime_proxy"],
                # Robustness & Expert settings (Phase 7)
                "shrinkage_enabled": preset_config["shrinkage_enabled"],
                "shrinkage_method": preset_config["shrinkage_method"],
                "leverage_cap": preset_config["leverage_cap"],
                "random_seed": preset_config["random_seed"],
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": preset_config["z_entry_soft"],
                "z_exit_soft": preset_config["z_exit_soft"],
                "soft_strikes": preset_config["soft_strikes"],
                "entry_soft_strikes": preset_config["entry_soft_strikes"],
                "sticky_add_periods": preset_config["sticky_add_periods"],
                "sticky_drop_periods": preset_config["sticky_drop_periods"],
                "ci_level": preset_config["ci_level"],
            }
            st.rerun()

    # Weighting scheme selector (outside form for dynamic description updates)
    st.markdown("---")
    st.subheader("📊 Weighting Scheme")
    weighting_labels = [label for label, _ in WEIGHTING_SCHEMES]
    weighting_values = [value for _, value in WEIGHTING_SCHEMES]
    current_weighting = model_state.get("weighting_scheme", "equal")
    try:
        weighting_index = weighting_values.index(current_weighting)
    except ValueError:
        weighting_index = 0

    weighting_value = st.selectbox(
        "Select Weighting Scheme",
        options=weighting_values,
        format_func=lambda x: weighting_labels[weighting_values.index(x)],
        index=weighting_index,
        help=HELP_TEXT["weighting"],
        key="weighting_scheme_selector",
    )

    # Show description for selected weighting scheme (updates dynamically)
    with st.expander("ℹ️ About this weighting scheme", expanded=False):
        st.markdown(
            WEIGHTING_DESCRIPTIONS.get(weighting_value, "No description available.")
        )

    # Update model_state if weighting changed
    if weighting_value != current_weighting:
        st.session_state["model_state"]["weighting_scheme"] = weighting_value

    with st.form("model_settings", clear_on_submit=False):
        # Section 1: Fund Selection Settings
        st.subheader("📋 Fund Selection Settings")
        st.caption(
            "Configure how funds are evaluated and filtered for portfolio inclusion."
        )

        c1, c2 = st.columns(2)
        with c1:
            lookback = st.number_input(
                "Lookback Window (months)",
                min_value=12,
                value=int(model_state.get("lookback_months", 36)),
                help=HELP_TEXT["lookback"],
            )
        with c2:
            min_history = st.number_input(
                "Minimum History Required (months)",
                min_value=1,
                value=int(
                    model_state.get(
                        "min_history_months", model_state.get("lookback_months", 36)
                    )
                ),
                help=HELP_TEXT["min_history"],
            )

        # Section 2: Portfolio Settings
        st.divider()
        st.subheader("📈 Portfolio Settings")
        st.caption("Configure how the portfolio is constructed from selected funds.")

        c3, c4 = st.columns(2)
        with c3:
            evaluation = st.number_input(
                "Evaluation Window (months)",
                min_value=3,
                value=int(model_state.get("evaluation_months", 12)),
                help=HELP_TEXT["evaluation"],
            )
        with c4:
            selection = st.number_input(
                "Selection Count",
                min_value=1,
                max_value=len(fund_cols) if fund_cols else 100,
                value=min(
                    int(model_state.get("selection_count", 10)),
                    len(fund_cols) if fund_cols else 10,
                ),
                help=HELP_TEXT["selection"],
            )

        # Section 3: Metric Weights
        st.divider()
        st.subheader("⚖️ Metric Weights")
        st.caption(
            "Relative importance of each metric when ranking funds for selection."
        )

        metric_weights: dict[str, float] = {}
        # Create two rows for the 6 metrics
        help_keys = [
            "sharpe_weight",
            "return_weight",
            "sortino_weight",
            "info_ratio_weight",
            "drawdown_weight",
            "vol_weight",
        ]

        # First row: 3 metrics
        col1, col2, col3 = st.columns(3)
        cols_row1 = [col1, col2, col3]
        for i in range(min(3, len(METRIC_FIELDS))):
            label, code = METRIC_FIELDS[i]
            help_key = help_keys[i]
            with cols_row1[i]:
                metric_weights[code] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(model_state.get("metric_weights", {}).get(code, 1.0)),
                    step=0.1,
                    help=HELP_TEXT.get(
                        help_key, "Weight for this metric in fund ranking."
                    ),
                    key=f"metric_{code}",
                )

        # Second row: remaining metrics
        if len(METRIC_FIELDS) > 3:
            col4, col5, col6 = st.columns(3)
            cols_row2 = [col4, col5, col6]
            for i in range(3, len(METRIC_FIELDS)):
                label, code = METRIC_FIELDS[i]
                help_key = help_keys[i] if i < len(help_keys) else "vol_weight"
                with cols_row2[i - 3]:
                    metric_weights[code] = st.number_input(
                        label,
                        min_value=0.0,
                        value=float(
                            model_state.get("metric_weights", {}).get(code, 1.0)
                        ),
                        step=0.1,
                        help=HELP_TEXT.get(
                            help_key, "Weight for this metric in fund ranking."
                        ),
                        key=f"metric_{code}",
                    )

        # Show weight sum
        weight_sum = sum(float(w or 0) for w in metric_weights.values())
        if weight_sum > 0:
            st.caption(
                f"📊 Total weight: {weight_sum:.2f} — Weights will be auto-normalized to sum to 1.0 during analysis."
            )
        else:
            st.warning("⚠️ Set at least one metric weight > 0.")

        # Benchmark selector for Info Ratio - always show when info_ratio weight > 0
        # Check both form value AND saved state for info_ratio weight
        info_ratio_weight = metric_weights.get("info_ratio", 0)
        saved_info_ratio_weight = model_state.get("metric_weights", {}).get(
            "info_ratio", 0
        )
        show_benchmark_selector = info_ratio_weight > 0 or saved_info_ratio_weight > 0

        info_ratio_benchmark = model_state.get("info_ratio_benchmark", "")
        if show_benchmark_selector:
            st.divider()
            st.markdown("**📈 Information Ratio Benchmark**")
            st.caption(
                "Select the index or benchmark column to use for Information Ratio calculation."
            )
            current_benchmark = model_state.get("info_ratio_benchmark", "")
            benchmark_index = 0
            if current_benchmark and current_benchmark in benchmark_options:
                benchmark_index = (
                    benchmark_options.index(current_benchmark) + 1
                )  # +1 for "Select..." option

            benchmark_selection = st.selectbox(
                "Benchmark Column",
                options=["(Select a benchmark)"] + benchmark_options,
                index=benchmark_index,
                help=HELP_TEXT["info_ratio_benchmark"],
                key="benchmark_selector",
            )
            if benchmark_selection != "(Select a benchmark)":
                info_ratio_benchmark = benchmark_selection
        else:
            info_ratio_benchmark = ""

        # Section 4: Risk Settings
        st.divider()
        st.subheader("🎯 Risk Settings")

        risk_c1, risk_c2 = st.columns(2)
        with risk_c1:
            risk_target = st.number_input(
                "Target Portfolio Volatility",
                min_value=0.01,
                max_value=0.50,
                value=float(model_state.get("risk_target", 0.1)),
                step=0.01,
                format="%.2f",
                help=HELP_TEXT["risk_target"],
            )
            st.caption(f"Target: {risk_target:.0%} annualized vol")

        with risk_c2:
            rf_rate_pct = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(model_state.get("rf_rate_annual", 0.0)) * 100,
                step=0.25,
                format="%.2f",
                help=HELP_TEXT["rf_rate"],
            )
            # Convert to decimal for storage
            rf_rate_annual = rf_rate_pct / 100.0
            st.caption(f"Used for Sharpe/Sortino: {rf_rate_pct:.2f}%")

        # Volatility floor and warmup
        vol_c1, vol_c2 = st.columns(2)
        with vol_c1:
            vol_floor_pct = st.number_input(
                "Volatility Floor (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(model_state.get("vol_floor", 0.015)) * 100,
                step=0.1,
                format="%.2f",
                help=HELP_TEXT["vol_floor"],
            )
            # Convert to decimal for storage
            vol_floor = vol_floor_pct / 100.0

        with vol_c2:
            warmup_periods = st.number_input(
                "Warmup Periods",
                min_value=0,
                max_value=24,
                value=int(model_state.get("warmup_periods", 0)),
                help=HELP_TEXT["warmup_periods"],
            )

        # Volatility Adjustment Settings (Phase 10) - collapsible
        with st.expander("📊 Volatility Adjustment Details", expanded=False):
            st.caption(
                "Configure how volatility scaling is applied to returns. "
                "These settings control the rolling window for volatility estimation."
            )

            vol_adj_enabled = st.checkbox(
                "Enable Volatility Adjustment",
                value=bool(model_state.get("vol_adjust_enabled", True)),
                help=HELP_TEXT["vol_adjust_enabled"],
            )

            if vol_adj_enabled:
                va_c1, va_c2, va_c3 = st.columns(3)
                with va_c1:
                    vol_window_length = st.number_input(
                        "Vol Window (periods)",
                        min_value=10,
                        max_value=252,
                        value=int(model_state.get("vol_window_length", 63)),
                        help=HELP_TEXT["vol_window_length"],
                    )
                    st.caption(
                        f"~{vol_window_length // 21} months"
                        if vol_window_length >= 21
                        else f"{vol_window_length} days"
                    )

                with va_c2:
                    decay_methods = ["ewma", "simple"]
                    decay_labels = {
                        "ewma": "EWMA (Exponential)",
                        "simple": "Simple (Equal Weight)",
                    }
                    current_decay = model_state.get("vol_window_decay", "ewma")
                    vol_window_decay = st.selectbox(
                        "Decay Method",
                        options=decay_methods,
                        format_func=lambda x: decay_labels.get(x, x),
                        index=(
                            decay_methods.index(current_decay)
                            if current_decay in decay_methods
                            else 0
                        ),
                        help=HELP_TEXT["vol_window_decay"],
                    )

                with va_c3:
                    vol_ewma_lambda = st.number_input(
                        "EWMA Lambda",
                        min_value=0.80,
                        max_value=0.99,
                        value=float(model_state.get("vol_ewma_lambda", 0.94)),
                        step=0.01,
                        format="%.2f",
                        help=HELP_TEXT["vol_ewma_lambda"],
                        disabled=(vol_window_decay != "ewma"),
                    )
                    if vol_window_decay == "ewma":
                        half_life = round(-1 / (1 + 1e-9 - vol_ewma_lambda), 1)
                        st.caption(f"Half-life: ~{half_life:.0f} periods")
            else:
                vol_window_length = int(model_state.get("vol_window_length", 63))
                vol_window_decay = model_state.get("vol_window_decay", "ewma")
                vol_ewma_lambda = float(model_state.get("vol_ewma_lambda", 0.94))

        max_weight = st.number_input(
            "Maximum Weight per Fund (%)",
            min_value=5,
            max_value=100,
            value=int(model_state.get("max_weight", 0.20) * 100),
            step=5,
            help=HELP_TEXT["max_weight"],
        )
        # Convert back to decimal for storage
        max_weight_decimal = max_weight / 100.0

        # Section 5: Advanced Settings
        st.divider()
        st.subheader("⚙️ Advanced Settings")
        st.caption("Fine-tune fund addition/removal rules and transaction costs.")

        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            cooldown_months = st.number_input(
                "Cooldown Period (months)",
                min_value=0,
                max_value=24,
                value=int(model_state.get("cooldown_months", 3)),
                help=HELP_TEXT["cooldown_months"],
            )
            min_track_months = st.number_input(
                "Minimum Track Record (months)",
                min_value=1,
                max_value=120,
                value=int(model_state.get("min_track_months", 24)),
                help=HELP_TEXT["min_track_months"],
            )

        with adv_c2:
            rebalance_options = ["M", "Q", "A"]
            rebalance_labels = {"M": "Monthly", "Q": "Quarterly", "A": "Annually"}
            current_rebal = model_state.get("rebalance_freq", "M")
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                options=rebalance_options,
                format_func=lambda x: rebalance_labels.get(x, x),
                index=(
                    rebalance_options.index(current_rebal)
                    if current_rebal in rebalance_options
                    else 0
                ),
                help=HELP_TEXT["rebalance_freq"],
            )
            max_turnover = st.number_input(
                "Maximum Turnover",
                min_value=0.0,
                max_value=2.0,
                value=float(model_state.get("max_turnover", 1.0)),
                step=0.1,
                format="%.1f",
                help=HELP_TEXT["max_turnover"],
            )

        transaction_cost_bps = st.number_input(
            "Transaction Cost (basis points)",
            min_value=0,
            max_value=100,
            value=int(model_state.get("transaction_cost_bps", 0)),
            help=HELP_TEXT["transaction_cost_bps"],
        )
        if transaction_cost_bps > 0:
            st.caption(
                f"Each trade incurs a {transaction_cost_bps} bp ({transaction_cost_bps/100:.2f}%) cost."
            )

        # Section 6: Fund Holding Rules (Phase 3)
        st.divider()
        st.subheader("🔒 Fund Holding Rules")
        st.caption("Control fund tenure and portfolio churn limits.")

        hold_c1, hold_c2, hold_c3 = st.columns(3)
        with hold_c1:
            min_tenure_periods = st.number_input(
                "Min Tenure (periods)",
                min_value=0,
                max_value=24,
                value=int(model_state.get("min_tenure_periods", 3)),
                help=HELP_TEXT["min_tenure"],
            )
            if min_tenure_periods > 0:
                st.caption(f"Funds held for at least {min_tenure_periods} periods")

        with hold_c2:
            max_changes_per_period = st.number_input(
                "Max Changes/Period",
                min_value=0,
                max_value=50,
                value=int(model_state.get("max_changes_per_period", 0)),
                help=HELP_TEXT["max_changes"],
            )
            if max_changes_per_period == 0:
                st.caption("Unlimited changes allowed")
            else:
                st.caption(f"Max {max_changes_per_period} adds/removes")

        with hold_c3:
            max_active_positions = st.number_input(
                "Max Active Positions",
                min_value=0,
                max_value=100,
                value=int(model_state.get("max_active_positions", 0)),
                help=HELP_TEXT["max_active"],
            )
            if max_active_positions == 0:
                st.caption("Uses selection count")
            else:
                st.caption(f"Capped at {max_active_positions} funds")

        # Section 7: Trend Signal Settings (Phase 4) - collapsible
        st.divider()
        with st.expander("📈 Trend Signal Settings (Advanced)", expanded=False):
            st.caption(
                "Configure the momentum signal generation parameters. "
                "These control how trend signals are computed for fund ranking."
            )

            sig_c1, sig_c2, sig_c3 = st.columns(3)
            with sig_c1:
                trend_window = st.number_input(
                    "Signal Window (periods)",
                    min_value=5,
                    max_value=252,
                    value=int(model_state.get("trend_window", 63)),
                    help=HELP_TEXT["trend_window"],
                )
                st.caption(
                    f"~{trend_window // 21} months"
                    if trend_window >= 21
                    else f"{trend_window} days"
                )

            with sig_c2:
                trend_lag = st.number_input(
                    "Signal Lag",
                    min_value=1,
                    max_value=10,
                    value=int(model_state.get("trend_lag", 1)),
                    help=HELP_TEXT["trend_lag"],
                )

            with sig_c3:
                trend_min_periods_val = model_state.get("trend_min_periods")
                trend_min_periods = st.number_input(
                    "Min Periods",
                    min_value=0,
                    max_value=252,
                    value=int(trend_min_periods_val) if trend_min_periods_val else 0,
                    help=HELP_TEXT["trend_min_periods"],
                )
                # Convert 0 to None for storage
                trend_min_periods_out = (
                    trend_min_periods if trend_min_periods > 0 else None
                )
                if trend_min_periods == 0:
                    st.caption("Uses full window")

            sig_c4, sig_c5 = st.columns(2)
            with sig_c4:
                trend_zscore = st.checkbox(
                    "Cross-sectional Z-score",
                    value=bool(model_state.get("trend_zscore", False)),
                    help=HELP_TEXT["trend_zscore"],
                )

            with sig_c5:
                trend_vol_adjust = st.checkbox(
                    "Volatility adjust signals",
                    value=bool(model_state.get("trend_vol_adjust", False)),
                    help=HELP_TEXT["trend_vol_adjust"],
                )

            # Show vol target only if vol adjust is enabled
            trend_vol_target_out = None
            if trend_vol_adjust:
                vol_tgt_val = model_state.get("trend_vol_target")
                trend_vol_target = st.number_input(
                    "Signal Vol Target",
                    min_value=0.01,
                    max_value=0.50,
                    value=float(vol_tgt_val) if vol_tgt_val else 0.10,
                    step=0.01,
                    format="%.2f",
                    help=HELP_TEXT["trend_vol_target"],
                )
                trend_vol_target_out = trend_vol_target
                st.caption(f"Target: {trend_vol_target:.0%} annualized")

        # Section 8: Regime Analysis (Phase 6) - collapsible
        st.divider()
        with st.expander("🔄 Regime Analysis (Advanced)", expanded=False):
            st.caption(
                "Enable regime detection to adjust portfolio behavior based on "
                "market conditions (risk-on vs risk-off)."
            )

            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                regime_enabled = st.checkbox(
                    "Enable Regime Detection",
                    value=bool(model_state.get("regime_enabled", False)),
                    help=HELP_TEXT["regime_enabled"],
                )

            with reg_c2:
                # Use benchmark columns as regime proxy options
                regime_proxy_options = ["SPX", "TSX", "MSCI", "ACWI"] + [
                    c
                    for c in benchmark_options
                    if c.upper() not in ["SPX", "TSX", "MSCI", "ACWI"]
                ][
                    :10
                ]  # Limit to 14 options
                current_regime_proxy = model_state.get("regime_proxy", "SPX")
                regime_proxy = st.selectbox(
                    "Regime Proxy Index",
                    options=regime_proxy_options,
                    index=(
                        regime_proxy_options.index(current_regime_proxy)
                        if current_regime_proxy in regime_proxy_options
                        else 0
                    ),
                    help=HELP_TEXT["regime_proxy"],
                    disabled=not regime_enabled,
                )

            if regime_enabled:
                st.info(
                    "📊 Regime detection will classify periods as risk-on or risk-off "
                    f"based on {regime_proxy} returns and volatility."
                )

        # Section 9: Expert Settings (Phase 7) - collapsible
        st.divider()
        with st.expander("🔧 Expert Settings", expanded=False):
            st.caption(
                "Advanced settings for covariance matrix handling, leverage limits, "
                "and reproducibility. Most users can leave these at defaults."
            )

            # Covariance shrinkage settings
            st.markdown("**Covariance Matrix Robustness**")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                shrinkage_enabled = st.checkbox(
                    "Enable Shrinkage",
                    value=bool(model_state.get("shrinkage_enabled", True)),
                    help=HELP_TEXT["shrinkage_enabled"],
                )

            with exp_c2:
                shrinkage_methods = ["ledoit_wolf", "oas", "none"]
                shrinkage_labels = {
                    "ledoit_wolf": "Ledoit-Wolf",
                    "oas": "Oracle Approximating (OAS)",
                    "none": "None (raw)",
                }
                current_shrinkage = model_state.get("shrinkage_method", "ledoit_wolf")
                shrinkage_method = st.selectbox(
                    "Shrinkage Method",
                    options=shrinkage_methods,
                    format_func=lambda x: shrinkage_labels.get(x, x),
                    index=(
                        shrinkage_methods.index(current_shrinkage)
                        if current_shrinkage in shrinkage_methods
                        else 0
                    ),
                    help=HELP_TEXT["shrinkage_method"],
                    disabled=not shrinkage_enabled,
                )

            # Phase 14: Robustness fallbacks
            st.markdown("**Numerical Stability & Fallback**")
            rob_c1, rob_c2 = st.columns(2)
            with rob_c1:
                condition_threshold = st.number_input(
                    "Condition Number Threshold",
                    min_value=1.0e6,
                    max_value=1.0e15,
                    value=float(model_state.get("condition_threshold", 1.0e12)),
                    format="%.0e",
                    help=HELP_TEXT["condition_threshold"],
                )
            with rob_c2:
                safe_modes = ["hrp", "risk_parity", "equal"]
                safe_mode_labels = {
                    "hrp": "HRP (Hierarchical Risk Parity)",
                    "risk_parity": "Risk Parity",
                    "equal": "Equal Weight",
                }
                current_safe_mode = model_state.get("safe_mode", "hrp")
                safe_mode = st.selectbox(
                    "Fallback Method",
                    options=safe_modes,
                    format_func=lambda x: safe_mode_labels.get(x, x),
                    index=(
                        safe_modes.index(current_safe_mode)
                        if current_safe_mode in safe_modes
                        else 0
                    ),
                    help=HELP_TEXT["safe_mode"],
                )
            st.caption(
                f"If condition number exceeds {condition_threshold:.0e}, "
                f"fallback to {safe_mode_labels.get(safe_mode, safe_mode)}."
            )

            # Leverage and seed
            st.markdown("**Portfolio Limits & Reproducibility**")
            exp_c3, exp_c4 = st.columns(2)
            with exp_c3:
                leverage_cap = st.number_input(
                    "Leverage Cap",
                    min_value=1.0,
                    max_value=5.0,
                    value=float(model_state.get("leverage_cap", 2.0)),
                    step=0.5,
                    format="%.1f",
                    help=HELP_TEXT["leverage_cap"],
                )
                st.caption(f"Max gross exposure: {leverage_cap:.1f}x")

            with exp_c4:
                random_seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=99999,
                    value=int(model_state.get("random_seed", 42)),
                    help=HELP_TEXT["random_seed"],
                )

            # Phase 15: Constraints
            st.markdown("**Constraints**")
            long_only = st.checkbox(
                "Long-Only Portfolio",
                value=bool(model_state.get("long_only", True)),
                help=HELP_TEXT["long_only"],
            )
            if not long_only:
                st.warning(
                    "⚠️ Short positions enabled. Ensure your data and strategy "
                    "support short selling."
                )

            # Phase 16: Data/Preprocessing
            st.markdown("**Data Preprocessing**")
            prep_c1, prep_c2 = st.columns(2)
            with prep_c1:
                missing_policies = ["drop", "ffill", "zero"]
                missing_labels = {
                    "drop": "Drop Missing",
                    "ffill": "Forward Fill",
                    "zero": "Replace with Zero",
                }
                current_missing = model_state.get("missing_policy", "ffill")
                missing_policy = st.selectbox(
                    "Missing Data Policy",
                    options=missing_policies,
                    format_func=lambda x: missing_labels.get(x, x),
                    index=(
                        missing_policies.index(current_missing)
                        if current_missing in missing_policies
                        else 1
                    ),
                    help=HELP_TEXT["missing_policy"],
                )
            with prep_c2:
                winsorize_enabled = st.checkbox(
                    "Enable Winsorization",
                    value=bool(model_state.get("winsorize_enabled", True)),
                    help=HELP_TEXT["winsorize_enabled"],
                )

            if winsorize_enabled:
                win_c1, win_c2 = st.columns(2)
                with win_c1:
                    winsorize_lower = st.number_input(
                        "Lower Limit (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(model_state.get("winsorize_lower", 1.0)),
                        step=0.5,
                        format="%.1f",
                        help=HELP_TEXT["winsorize_lower"],
                    )
                with win_c2:
                    winsorize_upper = st.number_input(
                        "Upper Limit (%)",
                        min_value=90.0,
                        max_value=100.0,
                        value=float(model_state.get("winsorize_upper", 99.0)),
                        step=0.5,
                        format="%.1f",
                        help=HELP_TEXT["winsorize_upper"],
                    )
                st.caption(
                    f"Extreme returns clipped to [{winsorize_lower:.1f}%, "
                    f"{winsorize_upper:.1f}%] percentiles."
                )
            else:
                winsorize_lower = float(model_state.get("winsorize_lower", 1.0))
                winsorize_upper = float(model_state.get("winsorize_upper", 99.0))

        # Section 10: Entry/Exit Rules (Phase 5) - collapsible
        st.divider()
        with st.expander("🚪 Entry/Exit Rules", expanded=False):
            st.caption(
                "Configure how funds are added to and removed from the portfolio. "
                "These settings control manager hiring and firing decisions."
            )

            st.markdown("**Z-Score Thresholds**")
            st.info(
                "Z-scores measure how a fund's performance compares to peers. "
                "Positive = above average, Negative = below average."
            )
            ee_c1, ee_c2 = st.columns(2)
            with ee_c1:
                z_entry_soft = st.number_input(
                    "Entry Threshold (Z-Score)",
                    min_value=-2.0,
                    max_value=3.0,
                    value=float(model_state.get("z_entry_soft", 1.0)),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_entry_soft"],
                )
                st.caption(
                    f"Fund must score ≥ {z_entry_soft:.2f}σ above average to be "
                    "considered for addition."
                )

            with ee_c2:
                z_exit_soft = st.number_input(
                    "Exit Threshold (Z-Score)",
                    min_value=-3.0,
                    max_value=1.0,
                    value=float(model_state.get("z_exit_soft", -1.0)),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_exit_soft"],
                )
                st.caption(
                    f"Fund scoring ≤ {z_exit_soft:.2f}σ below average may be "
                    "considered for removal."
                )

            # Phase 13: Hard thresholds (immediate action)
            st.markdown("**Hard Thresholds (Immediate Action)**")
            st.caption(
                "Hard thresholds trigger immediate action without waiting for "
                "consecutive strikes. Leave blank to disable."
            )
            hard_c1, hard_c2 = st.columns(2)
            with hard_c1:
                z_entry_hard_val = model_state.get("z_entry_hard")
                z_entry_hard_enabled = st.checkbox(
                    "Enable Hard Entry Threshold",
                    value=z_entry_hard_val is not None,
                )
                if z_entry_hard_enabled:
                    z_entry_hard = st.number_input(
                        "Hard Entry Threshold",
                        min_value=0.0,
                        max_value=4.0,
                        value=float(
                            z_entry_hard_val if z_entry_hard_val is not None else 2.0
                        ),
                        step=0.25,
                        format="%.2f",
                        help=HELP_TEXT["z_entry_hard"],
                    )
                    st.caption(f"Fund scoring ≥ {z_entry_hard:.2f}σ instantly added.")
                else:
                    z_entry_hard = None
            with hard_c2:
                z_exit_hard_val = model_state.get("z_exit_hard")
                z_exit_hard_enabled = st.checkbox(
                    "Enable Hard Exit Threshold",
                    value=z_exit_hard_val is not None,
                )
                if z_exit_hard_enabled:
                    z_exit_hard = st.number_input(
                        "Hard Exit Threshold",
                        min_value=-4.0,
                        max_value=0.0,
                        value=float(
                            z_exit_hard_val if z_exit_hard_val is not None else -2.0
                        ),
                        step=0.25,
                        format="%.2f",
                        help=HELP_TEXT["z_exit_hard"],
                    )
                    st.caption(f"Fund scoring ≤ {z_exit_hard:.2f}σ instantly removed.")
                else:
                    z_exit_hard = None

            st.markdown("**Consecutive Period Requirements**")
            ee_c3, ee_c4 = st.columns(2)
            with ee_c3:
                soft_strikes = st.number_input(
                    "Exit Strikes",
                    min_value=1,
                    max_value=6,
                    value=int(model_state.get("soft_strikes", 2)),
                    help=HELP_TEXT["soft_strikes"],
                )
                st.caption(
                    f"Fund must fail {soft_strikes} consecutive periods before removal."
                )

            with ee_c4:
                entry_soft_strikes = st.number_input(
                    "Entry Strikes",
                    min_value=1,
                    max_value=6,
                    value=int(model_state.get("entry_soft_strikes", 1)),
                    help=HELP_TEXT["entry_soft_strikes"],
                )
                st.caption(
                    f"Fund must pass {entry_soft_strikes} consecutive periods "
                    "before addition."
                )

            st.markdown("**Sticky Ranking (Policy Engine)**")
            ee_c5, ee_c6 = st.columns(2)
            with ee_c5:
                sticky_add_periods = st.number_input(
                    "Sticky Add Periods",
                    min_value=1,
                    max_value=6,
                    value=int(model_state.get("sticky_add_periods", 1)),
                    help=HELP_TEXT["sticky_add_periods"],
                )
                st.caption(
                    f"Fund must rank in top-K for {sticky_add_periods} periods "
                    "before hiring."
                )

            with ee_c6:
                sticky_drop_periods = st.number_input(
                    "Sticky Drop Periods",
                    min_value=1,
                    max_value=6,
                    value=int(model_state.get("sticky_drop_periods", 1)),
                    help=HELP_TEXT["sticky_drop_periods"],
                )
                st.caption(
                    f"Fund must rank in bottom-K for {sticky_drop_periods} periods "
                    "before firing."
                )

            st.markdown("**Confidence Interval Gate**")
            ci_level = st.slider(
                "Confidence Interval Level",
                min_value=0.0,
                max_value=0.99,
                value=float(model_state.get("ci_level", 0.0)),
                step=0.05,
                format="%.2f",
                help=HELP_TEXT["ci_level"],
            )
            if ci_level > 0:
                st.caption(
                    f"Fund entry requires {ci_level * 100:.0f}% confidence that "
                    "score exceeds threshold."
                )
            else:
                st.caption("Confidence interval gate is disabled.")

        # Section 11: Multi-Period & Selection Settings (Phase 8) - collapsible
        st.divider()
        with st.expander("📊 Multi-Period & Selection Settings", expanded=False):
            st.caption(
                "Configure rolling walk-forward analysis and fund selection approach. "
                "Multi-period analysis tests the strategy across multiple time windows."
            )

            # Multi-period toggle
            multi_period_enabled = st.checkbox(
                "Enable Multi-Period Walk-Forward Analysis",
                value=bool(model_state.get("multi_period_enabled", False)),
                help=HELP_TEXT["multi_period_enabled"],
            )

            if multi_period_enabled:
                st.markdown("**Rolling Window Settings**")
                mp_c1, mp_c2, mp_c3 = st.columns(3)
                with mp_c1:
                    multi_period_frequencies = ["M", "Q", "A"]
                    freq_labels = {
                        "M": "Monthly",
                        "Q": "Quarterly",
                        "A": "Annual",
                    }
                    current_mp_freq = model_state.get("multi_period_frequency", "A")
                    multi_period_frequency = st.selectbox(
                        "Period Frequency",
                        options=multi_period_frequencies,
                        format_func=lambda x: freq_labels.get(x, x),
                        index=(
                            multi_period_frequencies.index(current_mp_freq)
                            if current_mp_freq in multi_period_frequencies
                            else 2
                        ),
                        help=HELP_TEXT["multi_period_frequency"],
                    )

                with mp_c2:
                    in_sample_years = st.number_input(
                        "In-Sample Window (Years)",
                        min_value=1,
                        max_value=10,
                        value=int(model_state.get("in_sample_years", 3)),
                        help=HELP_TEXT["in_sample_years"],
                    )

                with mp_c3:
                    out_sample_years = st.number_input(
                        "Out-Sample Window (Years)",
                        min_value=1,
                        max_value=5,
                        value=int(model_state.get("out_sample_years", 1)),
                        help=HELP_TEXT["out_sample_years"],
                    )

                st.caption(
                    f"Strategy will be tested using {in_sample_years}-year training "
                    f"periods, evaluated on {out_sample_years}-year test periods, "
                    f"rolled forward {freq_labels[multi_period_frequency].lower()}."
                )

                # Multi-period bounds (Phase 12)
                st.markdown("**Fund Count Bounds**")
                bounds_c1, bounds_c2 = st.columns(2)
                with bounds_c1:
                    mp_min_funds = st.number_input(
                        "Minimum Funds",
                        min_value=1,
                        max_value=50,
                        value=int(model_state.get("mp_min_funds", 10)),
                        help=HELP_TEXT["mp_min_funds"],
                    )
                with bounds_c2:
                    mp_max_funds = st.number_input(
                        "Maximum Funds",
                        min_value=5,
                        max_value=100,
                        value=int(model_state.get("mp_max_funds", 25)),
                        help=HELP_TEXT["mp_max_funds"],
                    )
                if mp_min_funds > mp_max_funds:
                    st.warning(
                        "⚠️ Minimum funds exceeds maximum. Values will be swapped."
                    )
            else:
                multi_period_frequency = model_state.get("multi_period_frequency", "A")
                in_sample_years = int(model_state.get("in_sample_years", 3))
                out_sample_years = int(model_state.get("out_sample_years", 1))
                mp_min_funds = int(model_state.get("mp_min_funds", 10))
                mp_max_funds = int(model_state.get("mp_max_funds", 25))

            st.markdown("**Fund Selection Approach**")
            sel_c1, sel_c2 = st.columns(2)
            with sel_c1:
                inclusion_approaches = ["top_n", "top_pct", "threshold"]
                inclusion_labels = {
                    "top_n": "Top N Funds",
                    "top_pct": "Top Percentage",
                    "threshold": "Z-Score Threshold",
                }
                current_inclusion = model_state.get("inclusion_approach", "top_n")
                inclusion_approach = st.selectbox(
                    "Inclusion Approach",
                    options=inclusion_approaches,
                    format_func=lambda x: inclusion_labels.get(x, x),
                    index=(
                        inclusion_approaches.index(current_inclusion)
                        if current_inclusion in inclusion_approaches
                        else 0
                    ),
                    help=HELP_TEXT["inclusion_approach"],
                )

            with sel_c2:
                rank_transforms = ["none", "zscore", "rank"]
                transform_labels = {
                    "none": "None (Raw Scores)",
                    "zscore": "Z-Score Normalize",
                    "rank": "Percentile Rank",
                }
                current_transform = model_state.get("rank_transform", "none")
                rank_transform = st.selectbox(
                    "Score Transform",
                    options=rank_transforms,
                    format_func=lambda x: transform_labels.get(x, x),
                    index=(
                        rank_transforms.index(current_transform)
                        if current_transform in rank_transforms
                        else 0
                    ),
                    help=HELP_TEXT["rank_transform"],
                )

            # Conditional inputs based on inclusion approach (Phase 9)
            rank_pct = float(model_state.get("rank_pct", 0.10))
            rank_threshold = float(model_state.get("rank_threshold", 1.5))
            if inclusion_approach == "top_pct":
                rank_pct = st.number_input(
                    "Top Percentage",
                    min_value=0.01,
                    max_value=0.50,
                    value=rank_pct,
                    step=0.01,
                    format="%.2f",
                    help=HELP_TEXT["rank_pct"],
                )
                st.caption(f"Select top {rank_pct * 100:.0f}% of funds by score")
            elif inclusion_approach == "threshold":
                rank_threshold = st.number_input(
                    "Z-Score Threshold",
                    min_value=0.0,
                    max_value=3.0,
                    value=rank_threshold,
                    step=0.1,
                    format="%.1f",
                    help=HELP_TEXT["rank_threshold"],
                )
                st.caption(
                    f"Include funds with score ≥ {rank_threshold:.1f}σ above mean"
                )

            st.markdown("**Additional Cost & Exclusion Settings**")
            cost_c1, cost_c2 = st.columns(2)
            with cost_c1:
                slippage_bps = st.number_input(
                    "Slippage (bps)",
                    min_value=0,
                    max_value=50,
                    value=int(model_state.get("slippage_bps", 0)),
                    help=HELP_TEXT["slippage_bps"],
                )
                if slippage_bps > 0:
                    st.caption(
                        f"Additional {slippage_bps} bps ({slippage_bps / 100:.2f}%) "
                        "market impact cost per trade."
                    )

            with cost_c2:
                bottom_k = st.number_input(
                    "Exclude Bottom K Funds",
                    min_value=0,
                    max_value=10,
                    value=int(model_state.get("bottom_k", 0)),
                    help=HELP_TEXT["bottom_k"],
                )
                if bottom_k > 0:
                    st.caption(
                        f"Bottom {bottom_k} ranked funds will always be excluded."
                    )

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")

        if submitted:
            # Always set to Custom unless user explicitly selects Custom
            effective_preset = "Custom"

            candidate_state = {
                "preset": effective_preset,
                "lookback_months": lookback,
                "min_history_months": min_history,
                "evaluation_months": evaluation,
                "selection_count": selection,
                "weighting_scheme": weighting_value,
                "metric_weights": metric_weights,
                "risk_target": risk_target,
                "info_ratio_benchmark": info_ratio_benchmark,
                # Date settings (preserved from outside form)
                "date_mode": model_state.get("date_mode", "relative"),
                "start_date": model_state.get("start_date"),
                "end_date": model_state.get("end_date"),
                # Risk settings
                "rf_rate_annual": rf_rate_annual,
                "vol_floor": vol_floor,
                "warmup_periods": warmup_periods,
                # Advanced settings
                "max_weight": max_weight_decimal,
                "cooldown_months": cooldown_months,
                "min_track_months": min_track_months,
                "rebalance_freq": rebalance_freq,
                "max_turnover": max_turnover,
                "transaction_cost_bps": transaction_cost_bps,
                # Fund holding rules (Phase 3)
                "min_tenure_periods": min_tenure_periods,
                "max_changes_per_period": max_changes_per_period,
                "max_active_positions": max_active_positions,
                # Trend signal parameters (Phase 4)
                "trend_window": trend_window,
                "trend_lag": trend_lag,
                "trend_min_periods": trend_min_periods_out,
                "trend_zscore": trend_zscore,
                "trend_vol_adjust": trend_vol_adjust,
                "trend_vol_target": trend_vol_target_out,
                # Regime analysis (Phase 6)
                "regime_enabled": regime_enabled,
                "regime_proxy": regime_proxy,
                # Robustness & Expert settings (Phase 7)
                "shrinkage_enabled": shrinkage_enabled,
                "shrinkage_method": shrinkage_method,
                "leverage_cap": leverage_cap,
                "random_seed": random_seed,
                # Robustness fallbacks (Phase 14)
                "condition_threshold": condition_threshold,
                "safe_mode": safe_mode,
                # Constraints (Phase 15)
                "long_only": long_only,
                # Data/Preprocessing (Phase 16)
                "missing_policy": missing_policy,
                "winsorize_enabled": winsorize_enabled,
                "winsorize_lower": winsorize_lower,
                "winsorize_upper": winsorize_upper,
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": z_entry_soft,
                "z_exit_soft": z_exit_soft,
                "soft_strikes": soft_strikes,
                "entry_soft_strikes": entry_soft_strikes,
                "sticky_add_periods": sticky_add_periods,
                "sticky_drop_periods": sticky_drop_periods,
                "ci_level": ci_level,
                # Multi-period & Selection settings (Phase 8)
                "multi_period_enabled": multi_period_enabled,
                "multi_period_frequency": multi_period_frequency,
                "in_sample_years": in_sample_years,
                "out_sample_years": out_sample_years,
                "inclusion_approach": inclusion_approach,
                "rank_transform": rank_transform,
                "slippage_bps": slippage_bps,
                "bottom_k": bottom_k,
                # Selection approach details (Phase 9)
                "rank_pct": rank_pct,
                "rank_threshold": rank_threshold,
                # Multi-period bounds (Phase 12)
                "mp_min_funds": mp_min_funds,
                "mp_max_funds": mp_max_funds,
                # Hard thresholds (Phase 13)
                "z_entry_hard": z_entry_hard,
                "z_exit_hard": z_exit_hard,
            }
            errors = _validate_model(
                candidate_state, len(fund_cols) if fund_cols else 0
            )
            if errors:
                st.error("\n".join(f"• {err}" for err in errors))
            else:
                st.session_state["model_state"] = candidate_state
                analysis_runner.clear_cached_analysis()
                app_state.clear_analysis_results()
                st.success(
                    "✅ Model configuration saved. Go to Results to run analysis."
                )


render_model_page()
