"""Model configuration page for the Streamlit application."""

from __future__ import annotations

import difflib
import html
import json
from typing import Any, Mapping

import streamlit as st
import yaml

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner
from trend_analysis.config.patch import diff_configs

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


# Config chat panel helpers
def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric * 100:.1f}%"


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _config_summary_sections(
    model_state: Mapping[str, Any],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "Overview",
            [
                ("Preset", _format_value(model_state.get("preset"))),
                ("Weighting", _format_value(model_state.get("weighting_scheme"))),
                ("Selection count", _format_value(model_state.get("selection_count"))),
            ],
        ),
        (
            "Time Windows",
            [
                ("Lookback periods", _format_value(model_state.get("lookback_periods"))),
                ("Evaluation periods", _format_value(model_state.get("evaluation_periods"))),
                ("Min history", _format_value(model_state.get("min_history_periods"))),
                ("Frequency", _format_value(model_state.get("multi_period_frequency"))),
            ],
        ),
        (
            "Risk + Constraints",
            [
                ("Risk target", _format_percent(model_state.get("risk_target"))),
                ("Max weight", _format_percent(model_state.get("max_weight"))),
                ("Min weight", _format_percent(model_state.get("min_weight"))),
                ("Max turnover", _format_percent(model_state.get("max_turnover"))),
            ],
        ),
        (
            "Signals",
            [
                ("Trend window", _format_value(model_state.get("trend_window"))),
                ("Trend lag", _format_value(model_state.get("trend_lag"))),
                ("Vol adjust", _format_value(model_state.get("vol_adjust_enabled"))),
            ],
        ),
    ]


def _render_config_summary(model_state: Mapping[str, Any] | None) -> None:
    if not model_state:
        st.info("No configuration loaded yet.")
        return

    for title, rows in _config_summary_sections(model_state):
        st.markdown(f"**{title}**")
        for label, value in rows:
            st.markdown(f"- {label}: {value}")


def _render_diff_preview_styles() -> None:
    st.markdown(
        """
<style>
.config-diff {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  overflow-x: auto;
}
.config-diff .diff-line {
  padding: 2px 8px;
  white-space: pre;
}
.config-diff .diff-add { background: #e6ffed; color: #14532d; }
.config-diff .diff-remove { background: #ffeef0; color: #7f1d1d; }
.config-diff .diff-header { background: #f8fafc; color: #0f172a; font-weight: 600; }
.config-diff .diff-hunk { background: #eff6ff; color: #1d4ed8; }
.config-diff .diff-context { color: #111827; }
.config-diff-table table.diff {
  width: 100%;
  border-collapse: collapse;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.82rem;
  line-height: 1.35;
}
.config-diff-table table.diff th {
  background: #f8fafc;
  color: #0f172a;
  text-align: left;
  padding: 4px 6px;
}
.config-diff-table table.diff td {
  padding: 2px 6px;
  vertical-align: top;
  white-space: pre;
}
.config-diff-table .diff_add { background: #e6ffed; color: #14532d; }
.config-diff-table .diff_sub { background: #ffeef0; color: #7f1d1d; }
.config-diff-table .diff_chg { background: #fff7ed; color: #9a3412; }
</style>
""",
        unsafe_allow_html=True,
    )


def _diff_text_to_html(diff_text: str) -> str:
    lines = diff_text.splitlines()
    html_lines: list[str] = []
    for line in lines:
        if line.startswith(("+++ ", "--- ")):
            css_class = "diff-header"
        elif line.startswith("@@"):
            css_class = "diff-hunk"
        elif line.startswith("+"):
            css_class = "diff-add"
        elif line.startswith("-"):
            css_class = "diff-remove"
        else:
            css_class = "diff-context"
        safe_line = html.escape(line)
        html_lines.append(f'<div class="diff-line {css_class}">{safe_line}</div>')
    return '<div class="config-diff">' + "".join(html_lines) + "</div>"


def _render_unified_diff(diff_text: str) -> None:
    if not diff_text.strip():
        st.info("No differences found.")
        return
    _render_diff_preview_styles()
    st.markdown(_diff_text_to_html(diff_text), unsafe_allow_html=True)


def _render_side_by_side_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    before_yaml = yaml.safe_dump(dict(before), sort_keys=False, default_flow_style=False)
    after_yaml = yaml.safe_dump(dict(after), sort_keys=False, default_flow_style=False)
    differ = difflib.HtmlDiff(tabsize=2, wrapcolumn=80)
    diff_table = differ.make_table(
        before_yaml.splitlines(),
        after_yaml.splitlines(),
        fromdesc="Before",
        todesc="After",
        context=True,
        numlines=3,
    )
    _render_diff_preview_styles()
    st.markdown(
        f'<div class="config-diff config-diff-table">{diff_table}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Raw YAML", expanded=False):
        col_before, col_after = st.columns(2)
        with col_before:
            st.caption("Before")
            st.code(before_yaml, language="yaml")
        with col_after:
            st.caption("After")
            st.code(after_yaml, language="yaml")


def _render_config_diff_preview(model_state: Mapping[str, Any] | None) -> None:
    st.markdown("---")
    st.markdown("**Diff preview**")
    preview = st.session_state.get("config_chat_preview")
    if not isinstance(preview, Mapping):
        st.info("No preview available yet. Send an instruction to generate a diff.")
        return

    before = preview.get("before")
    if not isinstance(before, Mapping):
        before = model_state or {}
    after = preview.get("after")
    if not isinstance(after, Mapping):
        st.warning("Preview data is incomplete. Try generating a new diff.")
        return
    diff_text = preview.get("diff")
    if not isinstance(diff_text, str):
        diff_text = diff_configs(dict(before), dict(after))

    tabs = st.tabs(["Unified diff", "Side-by-side"])
    with tabs[0]:
        _render_unified_diff(diff_text)
    with tabs[1]:
        _render_side_by_side_diff(before, after)


def _render_config_chat_contents(model_state: Mapping[str, Any] | None) -> None:
    st.caption("Describe the configuration change you want to try.")
    instruction = st.text_area(
        "Instruction",
        key="config_chat_instruction",
        height=120,
        placeholder="e.g. Increase lookback to 24 months and reduce max weight to 10%",
    )
    send_clicked = st.button("Send", key="config_chat_send", use_container_width=True)
    if send_clicked:
        trimmed = instruction.strip()
        if not trimmed:
            st.warning("Enter an instruction before sending.")
        else:
            st.session_state["config_chat_last_instruction"] = trimmed
            st.success("Instruction captured. Preview coming next.")
    st.markdown("---")
    st.markdown("**Current configuration summary**")
    _render_config_summary(model_state)
    _render_config_diff_preview(model_state)


def render_config_chat_panel(
    *,
    location: str = "sidebar",
    model_state: Mapping[str, Any] | None = None,
) -> None:
    """Render the Config Chat panel for natural-language config tweaks."""

    if location == "sidebar":
        with st.sidebar:
            with st.expander("💬 Config Chat", expanded=False):
                _render_config_chat_contents(model_state)
        return

    with st.expander("💬 Config Chat", expanded=False):
        _render_config_chat_contents(model_state)


# Preset configurations with default parameter values
PRESET_CONFIGS = {
    "Baseline": {
        "lookback_periods": 3,
        "min_history_periods": 3,
        "evaluation_periods": 1,
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
        "rf_override_enabled": False,
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
        "min_weight": 0.05,
        "cooldown_periods": 1,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules (Phase 3)
        "min_tenure_periods": 3,
        "max_changes_per_period": 0,  # 0 = unlimited
        "max_active_positions": 0,  # 0 = unlimited (uses selection_count)
        # Portfolio signal parameters (Phase 4)
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
        "random_seed": 42,
        # Robustness fallbacks (Phase 14)
        "condition_threshold": 1.0e12,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": 1.0,
        "z_exit_soft": -1.0,
        "soft_strikes": 2,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 2,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "inclusion_approach": "threshold",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        # Multi-period bounds (Phase 12)
        "mp_min_funds": 10,
        "mp_max_funds": 25,
        # Hard thresholds (Phase 13)
        "z_entry_hard": None,
        "z_exit_hard": None,
    },
    "Conservative": {
        "lookback_periods": 5,
        "min_history_periods": 5,
        "evaluation_periods": 1,
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
        "min_weight": 0.05,
        "cooldown_periods": 2,
        "rebalance_freq": "Q",
        "max_turnover": 0.50,
        "transaction_cost_bps": 10,
        # Fund holding rules - conservative: higher tenure, limited changes
        "min_tenure_periods": 6,
        "max_changes_per_period": 2,
        "max_active_positions": 10,
        # Portfolio signal parameters - longer window for stability
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
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - conservative: stricter threshold
        "condition_threshold": 1.0e10,
        "safe_mode": "risk_parity",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds - conservative: stricter entry, lenient exit
        "z_entry_soft": 1.5,
        "z_exit_soft": -1.0,
        "soft_strikes": 3,
        "entry_soft_strikes": 2,
        "min_weight_strikes": 2,
        "sticky_add_periods": 2,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - conservative: longer periods
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "inclusion_approach": "threshold",
        "slippage_bps": 5,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.10,
        # Multi-period bounds (Phase 12) - conservative: narrower range
        "mp_min_funds": 8,
        "mp_max_funds": 15,
        # Hard thresholds (Phase 13) - conservative: enabled, stricter
        "z_entry_hard": 2.5,
        "z_exit_hard": -2.5,
    },
    "Aggressive": {
        "lookback_periods": 2,
        "min_history_periods": 2,
        "evaluation_periods": 1,
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
        "min_weight": 0.05,
        "cooldown_periods": 0,
        "rebalance_freq": "M",
        "max_turnover": 1.0,
        "transaction_cost_bps": 0,
        # Fund holding rules - aggressive: minimal constraints
        "min_tenure_periods": 1,
        "max_changes_per_period": 0,  # unlimited
        "max_active_positions": 0,  # unlimited
        # Portfolio signal parameters - shorter window for responsiveness
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
        "random_seed": 42,
        # Robustness fallbacks (Phase 14) - aggressive: higher tolerance
        "condition_threshold": 1.0e14,
        "safe_mode": "hrp",
        # Constraints (Phase 15)
        "long_only": True,
        # Entry/Exit thresholds - aggressive: lenient entry, quick exit
        "z_entry_soft": 0.5,
        "z_exit_soft": -0.5,
        "soft_strikes": 1,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 2,
        "sticky_add_periods": 1,
        "sticky_drop_periods": 1,
        "ci_level": 0.0,
        # Multi-period & Selection settings (Phase 8) - aggressive: shorter periods
        "multi_period_enabled": True,
        "multi_period_frequency": "Q",
        "inclusion_approach": "threshold",
        "slippage_bps": 0,
        "bottom_k": 0,
        # Selection approach details (Phase 9)
        "rank_pct": 0.15,  # more aggressive percentage
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
    "rf_override": "Override the risk-free rate from data with a constant value. ⚠️ Using a constant rate reduces accuracy vs. time-varying rates.",
    "rf_rate": "Constant annual risk-free rate fallback. Only used when override is enabled and no RF column is in the data.",
    "vol_floor": "Minimum volatility floor for scaling. Prevents extreme weights on low-vol assets.",
    "warmup_periods": "Initial periods where returns are zeroed out to allow volatility estimates to stabilize before calculating performance metrics.",
    # Phase 10: Volatility adjustment details
    "vol_adjust_enabled": "Enable volatility adjustment to scale returns to target vol.",
    "vol_window_length": "Rolling window for volatility estimation (periods). ~63 = 3 months.",
    "vol_window_decay": "EWMA weights recent data more; Simple uses equal weights.",
    "vol_ewma_lambda": "EWMA decay factor. Higher = longer memory. 0.94 is RiskMetrics standard.",
    # Advanced settings
    "max_weight": "Maximum allocation to any single fund. Prevents concentration risk.",
    "min_weight": "Minimum allocation per fund. Used as a weight floor and for underweight exit detection.",
    "cooldown_periods": "After a fund is removed, it cannot be re-added for this many periods.",
    "rebalance_freq": "How often to rebalance the portfolio weights.",
    "max_turnover": "Maximum portfolio turnover allowed per rebalance (1.0 = 100%).",
    "transaction_cost_bps": "Transaction cost in basis points (0.01% = 1 bp) applied per trade.",
    # Phase 3: Fund holding rules
    "min_tenure": "Minimum periods a fund must be held before it can be removed.",
    "max_changes": "Maximum number of fund additions/removals per rebalance. 0 = unlimited.",
    "max_active": "Maximum active positions in portfolio. 0 = use selection count.",
    # Phase 4: Trend signal parameters
    "trend_window": "Rolling window size for computing trend signals (in periods).",
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
    "random_seed": "Random seed for reproducibility. Change for different random selections.",
    # Phase 5: Entry/Exit thresholds
    "z_entry_soft": "Z-score threshold for fund entry consideration. Higher = stricter entry.",
    "z_exit_soft": "Z-score threshold for fund exit consideration. Lower = stricter exit.",
    "soft_strikes": "Consecutive periods below exit threshold before removing a fund.",
    "entry_soft_strikes": "Consecutive periods above entry threshold before adding a fund.",
    "min_weight_strikes": "Underweight exit: consecutive periods a fund's natural weight stays below the minimum weight before it is replaced. 0 = disable.",
    "sticky_add_periods": "Periods a fund must rank highly before being added to portfolio.",
    "sticky_drop_periods": "Periods a fund must rank poorly before being removed from portfolio.",
    "ci_level": "Confidence interval level for reporting only (0 = disabled, 0.9 = 90% CI).",
    # Phase 8: Multi-period & Selection settings
    "multi_period_enabled": "Enable rolling multi-period walk-forward analysis.",
    "multi_period_frequency": "Period frequency: Monthly (M), Quarterly (Q), or Annual (A).",
    "lookback_periods": "Number of periods for in-sample (training) window.",
    "evaluation_periods": "Number of periods for out-of-sample (testing) window.",
    "inclusion_approach": "How to select funds: Top N, Top Percentage, Z-score Threshold, Random, or Buy & Hold.",
    "buy_hold_initial": "Initial selection method for Buy & Hold mode.",
    "slippage_bps": "Additional slippage cost in basis points (market impact).",
    "bottom_k": "Number of bottom-ranked funds to always exclude (0 = none).",
    # Phase 9: Selection approach details
    "rank_pct": "Percentage of funds to include (0.10 = top 10%). Used with Top Percentage approach.",
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
    "long_only": (
        "Enforce long-only positions (no short selling). Built-in schemes (equal, "
        "score-prop, risk parity, HRP, ERC, robust_* defaults) are already non-negative "
        "unless you explicitly allow shorts (e.g., robust_mv with min_weight < 0). "
        "This matters when custom/manual weights or plugin engines allow shorts."
    ),
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
    lookback = values.get("lookback_periods", 3)
    min_history = values.get("min_history_periods", lookback)
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
        "lookback_periods": baseline["lookback_periods"],
        "min_history_periods": baseline["min_history_periods"],
        "evaluation_periods": baseline["evaluation_periods"],
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
        "min_weight": baseline.get("min_weight", 0.05),
        "cooldown_periods": baseline["cooldown_periods"],
        "rebalance_freq": baseline["rebalance_freq"],
        "max_turnover": baseline["max_turnover"],
        "transaction_cost_bps": baseline["transaction_cost_bps"],
        # Fund holding rules (Phase 3)
        "min_tenure_periods": baseline["min_tenure_periods"],
        "max_changes_per_period": baseline["max_changes_per_period"],
        "max_active_positions": baseline["max_active_positions"],
        # Portfolio signal parameters (Phase 4)
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
        "random_seed": baseline["random_seed"],
        # Robustness fallbacks (Phase 14)
        "condition_threshold": baseline["condition_threshold"],
        "safe_mode": baseline["safe_mode"],
        # Constraints (Phase 15)
        "long_only": baseline["long_only"],
        # Entry/Exit thresholds (Phase 5)
        "z_entry_soft": baseline["z_entry_soft"],
        "z_exit_soft": baseline["z_exit_soft"],
        "soft_strikes": baseline["soft_strikes"],
        "entry_soft_strikes": baseline["entry_soft_strikes"],
        "min_weight_strikes": baseline.get("min_weight_strikes", 2),
        "sticky_add_periods": baseline["sticky_add_periods"],
        "sticky_drop_periods": baseline["sticky_drop_periods"],
        "ci_level": baseline["ci_level"],
        # Multi-period & Selection settings (Phase 8)
        "multi_period_enabled": baseline["multi_period_enabled"],
        "multi_period_frequency": baseline["multi_period_frequency"],
        "inclusion_approach": baseline["inclusion_approach"],
        "slippage_bps": baseline["slippage_bps"],
        "bottom_k": baseline["bottom_k"],
        # Selection approach details (Phase 9)
        "rank_pct": baseline["rank_pct"],
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
    model_state = st.session_state.setdefault("model_state", _initial_model_state())
    render_config_chat_panel(model_state=model_state)
    st.title("Model Configuration")

    # Clarify this is for custom analysis
    st.info(
        "💡 This page is for **custom analysis** with your own data. "
        "For quick demos with preset configurations, use the **Run Demo** button on the Home page."
    )

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
        selected_rf = st.session_state.get("selected_risk_free")
        selected_bench = st.session_state.get("selected_benchmark")
        system_cols = {selected_rf, selected_bench, "Date", "DATE"} - {None}

        applied_funds = st.session_state.get("analysis_fund_columns")
        if not isinstance(applied_funds, list):
            applied_funds = st.session_state.get("fund_columns")

        if isinstance(applied_funds, list) and applied_funds:
            fund_cols = [c for c in applied_funds if c in df.columns and c not in system_cols]
        else:
            # Fallback: count only fund columns (exclude benchmarks/indices)
            fund_cols = [
                c
                for c in df.columns
                if c not in system_cols
                and c.upper()
                not in (
                    "SPX",
                    "TSX",
                    "RF",
                    "CASH",
                    "TBILL",
                    "TBILLS",
                    "T-BILL",
                )
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

    saved_model_states = app_state.get_saved_model_states()
    saved_names = sorted(saved_model_states)

    st.subheader("💾 Saved Configurations")
    with st.expander("Save, load, and manage model configurations", expanded=False):
        save_col, manage_col = st.columns(2)

        with save_col:
            st.markdown("**Save current settings**")
            with st.form("save_model_state_form"):
                default_name = st.session_state.get("active_saved_model_name", "")
                save_name = st.text_input(
                    "Configuration name",
                    value=default_name,
                    help="Provide a name to save the current model settings.",
                )
                overwrite_required = save_name.strip() in saved_model_states
                overwrite_confirmed = st.checkbox(
                    "Confirm overwrite",
                    value=False,
                    disabled=not overwrite_required,
                    help=(
                        "Required because a configuration with this name already exists."
                        if overwrite_required
                        else "Disabled until a duplicate name is entered."
                    ),
                )
                save_clicked = st.form_submit_button("Save Current Settings", type="primary")

            if save_clicked:
                trimmed = save_name.strip()
                if not trimmed:
                    st.error("Enter a name to save your configuration.")
                elif overwrite_required and not overwrite_confirmed:
                    st.warning("This name already exists. Check 'Confirm overwrite' to replace it.")
                else:
                    app_state.save_model_state(trimmed, st.session_state["model_state"])
                    st.session_state["active_saved_model_name"] = trimmed
                    st.success(f"Saved configuration '{trimmed}'.")
                    st.rerun()

        with manage_col:
            st.markdown("**Load or manage saved configurations**")
            if not saved_names:
                st.info("No saved configurations yet. Save one to enable loading and export.")
            else:
                selected_index = 0
                active_saved_name = st.session_state.get("active_saved_model_name")
                if active_saved_name in saved_names:
                    selected_index = saved_names.index(active_saved_name)
                selected_saved = st.selectbox(
                    "Saved configurations",
                    saved_names,
                    index=selected_index,
                    key="saved_configuration_selector",
                )

                if st.button("Load selected configuration", key="load_saved_config_button"):
                    st.session_state["model_state"] = app_state.load_saved_model_state(
                        selected_saved
                    )
                    st.session_state["active_saved_model_name"] = selected_saved
                    analysis_runner.clear_cached_analysis()
                    app_state.clear_analysis_results()
                    st.success(
                        f"Loaded configuration '{selected_saved}'. The form has been updated."
                    )
                    st.rerun()

                with st.form("rename_saved_config_form"):
                    rename_target = st.text_input(
                        "Rename selected configuration",
                        value=selected_saved,
                        key="rename_saved_config_input",
                    )
                    rename_clicked = st.form_submit_button("Rename configuration")

                if rename_clicked:
                    try:
                        app_state.rename_saved_model_state(selected_saved, rename_target)
                    except (KeyError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["active_saved_model_name"] = rename_target.strip()
                        st.success(f"Renamed configuration to '{rename_target.strip()}'.")
                        st.rerun()

                if st.button(
                    "Delete selected configuration",
                    key="delete_saved_config_button",
                    type="secondary",
                ):
                    app_state.delete_saved_model_state(selected_saved)
                    if st.session_state.get("active_saved_model_name") == selected_saved:
                        st.session_state.pop("active_saved_model_name", None)
                    st.success(f"Deleted configuration '{selected_saved}'.")
                    st.rerun()

        st.markdown("---")
        export_col, import_col = st.columns(2)
        with export_col:
            st.markdown("**Export saved configuration**")
            if saved_names:
                export_index = 0
                if st.session_state.get("active_saved_model_name") in saved_names:
                    export_index = saved_names.index(st.session_state["active_saved_model_name"])
                export_target = st.selectbox(
                    "Choose configuration to export",
                    saved_names,
                    index=export_index,
                    key="export_config_selector",
                )
                export_payload = app_state.export_model_state(export_target)
                st.text_area(
                    "Exported JSON",
                    value=export_payload,
                    height=160,
                    key="exported_config_payload",
                    help="Copy this JSON to share or reuse the configuration.",
                )
            else:
                st.info("Save a configuration to enable export.")

        with import_col:
            st.markdown("**Import configuration from JSON**")
            import_name = st.text_input("Name for imported configuration", key="import_config_name")
            import_payload = st.text_area("Paste JSON to import", key="import_config_payload")
            if st.button("Import JSON configuration", key="import_config_button"):
                if not import_payload.strip():
                    st.error("Paste a JSON payload to import a configuration.")
                else:
                    try:
                        imported_state = app_state.import_model_state(import_name, import_payload)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["active_saved_model_name"] = import_name.strip()
                        st.session_state["model_state"] = imported_state
                        analysis_runner.clear_cached_analysis()
                        app_state.clear_analysis_results()
                        st.success(
                            f"Imported configuration '{import_name.strip()}'. The form has been updated."
                        )
                        st.rerun()

        st.markdown("---")
        st.markdown("**Compare saved configurations**")
        if len(saved_names) < 2:
            st.info("Save at least two configurations to compare differences.")
        else:
            compare_col_a, compare_col_b = st.columns(2)
            with compare_col_a:
                config_a_name = st.selectbox(
                    "Configuration A",
                    saved_names,
                    index=0,
                    key="compare_config_a",
                )
            with compare_col_b:
                default_b = 1 if len(saved_names) > 1 else 0
                config_b_name = st.selectbox(
                    "Configuration B",
                    saved_names,
                    index=default_b,
                    key="compare_config_b",
                )

            if config_a_name == config_b_name:
                st.warning("Select two different configurations to compare.")
            else:
                diffs = app_state.diff_model_states(
                    saved_model_states[config_a_name],
                    saved_model_states[config_b_name],
                )
                if not diffs:
                    st.success("No differences found. The selected configurations match.")
                else:
                    diff_rows = []
                    for entry in diffs:
                        diff_rows.append(
                            {
                                "Setting": entry.path,
                                "Config A": (
                                    json.dumps(entry.left, sort_keys=True, default=str)
                                    if entry.left is not None
                                    else "—"
                                ),
                                "Config B": (
                                    json.dumps(entry.right, sort_keys=True, default=str)
                                    if entry.right is not None
                                    else "—"
                                ),
                                "Change": entry.change_type
                                + (" (type changed)" if entry.type_changed else ""),
                            }
                        )

                    st.dataframe(diff_rows, use_container_width=True, hide_index=True)

                    diff_text = app_state.format_model_state_diff(
                        diffs, label_a=config_a_name, label_b=config_b_name
                    )
                    st.caption("Copyable diff output:")
                    st.code(diff_text, language="text")

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
        # Convert data boundaries to date objects for the date picker
        if data_start is not None and hasattr(data_start, "date"):
            min_date = data_start.date()
        else:
            min_date = None

        if data_end is not None and hasattr(data_end, "date"):
            max_date = data_end.date()
        else:
            max_date = None

        # Show valid data range prominently
        if min_date and max_date:
            st.info(
                f"📅 **Available data range:** {min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')}"
            )

        date_col1, date_col2 = st.columns(2)
        # Get current values from model state
        current_start = model_state.get("start_date")
        current_end = model_state.get("end_date")

        # Track if dates were auto-corrected
        start_was_corrected = False
        end_was_corrected = False
        original_start_str = None
        original_end_str = None

        # Convert to date objects if they're strings
        if isinstance(current_start, str) and current_start:
            try:
                import datetime

                original_start_str = current_start
                current_start = datetime.datetime.strptime(
                    current_start[:7] + "-01", "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                current_start = min_date
        elif current_start is None:
            current_start = min_date

        # Ensure current_start is within valid range
        if current_start is not None and min_date is not None:
            if current_start < min_date:
                start_was_corrected = True
                current_start = min_date
        if current_start is not None and max_date is not None:
            if current_start > max_date:
                start_was_corrected = True
                current_start = max_date

        if isinstance(current_end, str) and current_end:
            try:
                import datetime

                original_end_str = current_end
                current_end = datetime.datetime.strptime(current_end[:7] + "-01", "%Y-%m-%d").date()
            except (ValueError, TypeError):
                current_end = max_date
        elif current_end is None:
            current_end = max_date

        # Ensure current_end is within valid range
        if current_end is not None and min_date is not None:
            if current_end < min_date:
                end_was_corrected = True
                current_end = min_date
        if current_end is not None and max_date is not None:
            if current_end > max_date:
                end_was_corrected = True
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
            # Show warning if date was auto-corrected
            if start_was_corrected and original_start_str:
                st.caption(f"⚠️ Adjusted from {original_start_str[:10]} to nearest available date")
            # Update model state
            if sim_start_date:
                st.session_state["model_state"]["start_date"] = sim_start_date.strftime("%Y-%m-%d")

        with date_col2:
            sim_end_date = st.date_input(
                "Simulation End Date",
                value=current_end,
                min_value=min_date,
                max_value=max_date,
                help=HELP_TEXT["end_date"],
                key="sim_end_date",
            )
            # Show warning if date was auto-corrected
            if end_was_corrected and original_end_str:
                st.caption(f"⚠️ Adjusted from {original_end_str[:10]} to nearest available date")
            # Update model state
            if sim_end_date:
                st.session_state["model_state"]["end_date"] = sim_end_date.strftime("%Y-%m-%d")

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
                    st.warning("Date range exceeds 50 years - please verify your selection.")
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
                "lookback_periods": preset_config["lookback_periods"],
                "min_history_periods": preset_config["min_history_periods"],
                "evaluation_periods": preset_config["evaluation_periods"],
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
                "min_weight": preset_config.get("min_weight", 0.05),
                "cooldown_periods": preset_config["cooldown_periods"],
                "rebalance_freq": preset_config["rebalance_freq"],
                "max_turnover": preset_config["max_turnover"],
                "transaction_cost_bps": preset_config["transaction_cost_bps"],
                # Fund holding rules (Phase 3)
                "min_tenure_periods": preset_config["min_tenure_periods"],
                "max_changes_per_period": preset_config["max_changes_per_period"],
                "max_active_positions": preset_config["max_active_positions"],
                # Portfolio signal parameters (Phase 4)
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
                "random_seed": preset_config["random_seed"],
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": preset_config["z_entry_soft"],
                "z_exit_soft": preset_config["z_exit_soft"],
                "soft_strikes": preset_config["soft_strikes"],
                "entry_soft_strikes": preset_config["entry_soft_strikes"],
                "min_weight_strikes": preset_config.get("min_weight_strikes", 2),
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
        st.markdown(WEIGHTING_DESCRIPTIONS.get(weighting_value, "No description available."))

    # Update model_state if weighting changed
    if weighting_value != current_weighting:
        st.session_state["model_state"]["weighting_scheme"] = weighting_value

    with st.form("model_settings", clear_on_submit=False):
        # =====================================================================
        # Section 0: Analysis Mode (Primary Choice)
        # =====================================================================
        st.subheader("🎯 Analysis Mode")
        st.caption(
            "Choose between single-period (one-time selection) or multi-period "
            "(rolling walk-forward analysis with rebalancing)."
        )

        multi_period_enabled = st.checkbox(
            "Enable Multi-Period Walk-Forward Analysis",
            value=bool(model_state.get("multi_period_enabled", True)),
            help=HELP_TEXT["multi_period_enabled"],
        )
        if multi_period_enabled:
            st.success(
                "✅ Funds will be re-evaluated at each period. Entry/exit rules and rebalancing apply."
            )

        # Fund Selection Approach - determines how funds are chosen
        st.markdown("**Fund Selection Approach**")
        approach_c1, approach_c2 = st.columns(2)
        with approach_c1:
            inclusion_approaches = [
                "top_n",
                "top_pct",
                "threshold",
                "random",
                "buy_and_hold",
            ]
            inclusion_labels = {
                "top_n": "Top N Funds (Ranking)",
                "top_pct": "Top Percentage (Ranking)",
                "threshold": "Z-Score Threshold",
                "random": "Random Selection",
                "buy_and_hold": "Buy & Hold",
            }
            current_inclusion = model_state.get("inclusion_approach", "threshold")
            inclusion_approach = st.selectbox(
                "Selection Method",
                options=inclusion_approaches,
                format_func=lambda x: inclusion_labels.get(x, x),
                index=(
                    inclusion_approaches.index(current_inclusion)
                    if current_inclusion in inclusion_approaches
                    else 2  # Default to "threshold" (index 2)
                ),
                help=HELP_TEXT["inclusion_approach"],
                key="inclusion_approach_select",
            )

        # Indicate whether this is ranking-based, threshold-based, random, or buy_and_hold
        is_ranking_mode = inclusion_approach in ["top_n", "top_pct"]
        is_random_mode = inclusion_approach == "random"
        is_top_n_mode = inclusion_approach == "top_n"
        is_top_pct_mode = inclusion_approach == "top_pct"
        is_buy_and_hold_mode = inclusion_approach == "buy_and_hold"

        # Buy & Hold initial selection method
        buy_hold_initial = "top_n"  # Default
        with approach_c2:
            if is_buy_and_hold_mode:
                # Show initial selection method for buy & hold
                buy_hold_options = ["top_n", "top_pct", "threshold", "random"]
                buy_hold_labels = {
                    "top_n": "Top N (Ranking)",
                    "top_pct": "Top Percentage",
                    "threshold": "Z-Score Threshold",
                    "random": "Random",
                }
                current_buy_hold = model_state.get("buy_hold_initial", "top_n")
                buy_hold_initial = st.selectbox(
                    "Initial Selection Method",
                    options=buy_hold_options,
                    format_func=lambda x: buy_hold_labels.get(x, x),
                    index=(
                        buy_hold_options.index(current_buy_hold)
                        if current_buy_hold in buy_hold_options
                        else 0
                    ),
                    help=HELP_TEXT.get(
                        "buy_hold_initial",
                        "How to select funds initially. Funds are held until they cease to exist.",
                    ),
                    key="buy_hold_initial_select",
                )
                st.caption(
                    f"🔒 **Buy & Hold**: Select funds using {buy_hold_labels[buy_hold_initial]}, "
                    "then hold until fund data disappears. Replacements use same method."
                )
            elif is_top_pct_mode:
                # Show percentage selector directly for top_pct mode
                rank_pct_input = st.number_input(
                    "Top Percentage (%)",
                    min_value=1,
                    max_value=50,
                    value=int(float(model_state.get("rank_pct", 0.10)) * 100),
                    step=1,
                    help="Select top N% of funds by score (e.g., 10 = top 10%)",
                    key="rank_pct_primary",
                )
                st.caption(f"🏆 Select top {rank_pct_input}% of funds by score")
            elif is_random_mode:
                st.caption(
                    "🎲 **Random Mode**: Funds are randomly selected each period. "
                    "No in-sample ranking metrics used for selection."
                )
            elif is_ranking_mode:
                st.caption(
                    "🏆 **Ranking Mode**: Funds are ranked by score and the top performers "
                    "are selected. Entry/exit uses ranking stability."
                )
            else:
                st.caption(
                    "📊 **Threshold Mode**: Funds must exceed a z-score threshold to enter. "
                    "Entry/exit uses z-score thresholds."
                )

        # =====================================================================
        # Section 1: Fund Selection Settings
        # =====================================================================
        st.divider()
        st.subheader("📋 Fund Selection & Time Windows")
        st.caption("Configure time windows for fund evaluation and walk-forward analysis.")

        # Row 1: Frequency (sets the period unit for all time windows)
        # Note: This is inside the form, so labels won't update until form is submitted.
        # We use model_state to determine the current unit for display.
        multi_period_frequencies = ["M", "Q", "A"]
        freq_labels = {
            "M": "Monthly",
            "Q": "Quarterly",
            "A": "Annual",
        }
        freq_period_labels = {
            "M": "months",
            "Q": "quarters",
            "A": "years",
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
        # Use the saved frequency from model_state for label display (shows current saved value)
        # The new selection will take effect after save

        # Row 2: Time windows
        st.markdown("**Time Windows**")
        c1, c2, c3 = st.columns(3)
        with c1:
            lookback = st.number_input(
                "Lookback",
                min_value=1,
                max_value=20,
                value=int(model_state.get("lookback_periods", 3)),
                help=HELP_TEXT.get(
                    "lookback_periods",
                    "Number of periods for in-sample (training) window.",
                ),
            )
            st.caption("In-sample history for ranking")
        with c2:
            evaluation = st.number_input(
                "Evaluation",
                min_value=1,
                max_value=10,
                value=int(model_state.get("evaluation_periods", 1)),
                help=HELP_TEXT.get(
                    "evaluation_periods",
                    "Number of periods for out-of-sample (testing) window.",
                ),
            )
            st.caption("Out-of-sample test period")
        with c3:
            min_history = st.number_input(
                "Min History",
                min_value=1,
                max_value=20,
                value=int(
                    model_state.get("min_history_periods", model_state.get("lookback_periods", 3))
                ),
                help=HELP_TEXT.get(
                    "min_history",
                    "Minimum periods of data required for a fund to be considered.",
                ),
            )
            st.caption("Funds with less history excluded")

        # Show period summary based on selected frequency
        selected_unit = freq_period_labels.get(multi_period_frequency, "periods")
        st.caption(
            f"Strategy: {lookback} {selected_unit} training → {evaluation} {selected_unit} testing, "
            f"rebalanced {freq_labels[multi_period_frequency].lower()}."
        )

        # Section 2: Portfolio Settings
        st.divider()
        st.subheader("📈 Portfolio Settings")
        st.caption("Configure how the portfolio is constructed from selected funds.")

        # Used by multiple controls: multi-period toggles are stored in model_state
        mp_enabled_state = bool(model_state.get("multi_period_enabled", True))

        st.markdown("**Portfolio size (target / min / max)**")
        size_c1, size_c2, size_c3 = st.columns(3)
        with size_c1:
            selection = st.number_input(
                "Target Funds (Initial N)",
                min_value=1,
                max_value=len(fund_cols) if fund_cols else 100,
                value=min(
                    int(model_state.get("selection_count", 10)),
                    len(fund_cols) if fund_cols else 10,
                ),
                help=(
                    "Target number of funds to hold (initial selection size). "
                    "Other constraints (min/max) may expand or cap holdings."
                ),
            )
            st.caption("Target holdings (initial selection size)")

        # Disable min/max for top_n mode since fund count is fixed by selection_count
        min_max_disabled = not mp_enabled_state or is_top_n_mode
        min_max_help_suffix = (
            " (disabled for Top N mode - uses Target Funds)" if is_top_n_mode else ""
        )

        with size_c2:
            mp_min_funds = st.number_input(
                "Minimum Funds",
                min_value=0,
                max_value=len(fund_cols) if fund_cols else 100,
                value=int(model_state.get("mp_min_funds", 0)),
                help=HELP_TEXT["mp_min_funds"] + min_max_help_suffix,
                disabled=min_max_disabled,
                key="mp_min_funds_input",
            )
            if is_top_n_mode:
                st.caption("🔒 Disabled (Top N uses Target Funds)")
            else:
                st.caption("Floor (0 = disabled)")

        with size_c3:
            mp_max_funds = st.number_input(
                "Maximum Funds",
                min_value=0,
                max_value=len(fund_cols) if fund_cols else 100,
                value=int(model_state.get("mp_max_funds", 0)),
                help=HELP_TEXT["mp_max_funds"] + min_max_help_suffix,
                disabled=min_max_disabled,
                key="mp_max_funds_input",
            )
            if is_top_n_mode:
                st.caption("🔒 Disabled (Top N uses Target Funds)")
            else:
                st.caption("Cap (0 = disabled)")

        st.markdown("**Portfolio weight constraints**")
        w_c1, w_c2 = st.columns(2)
        with w_c1:
            min_weight_pct = st.number_input(
                "Portfolio Weight Minimum per Fund (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(model_state.get("min_weight", 0.05)) * 100,
                step=0.5,
                format="%.2f",
                help=HELP_TEXT["min_weight"],
            )
            min_weight_decimal = min_weight_pct / 100.0

        with w_c2:
            max_weight = st.number_input(
                "Maximum Weight per Fund (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(model_state.get("max_weight", 0.20)) * 100,
                step=1.0,
                format="%.2f",
                help=HELP_TEXT["max_weight"],
            )
            max_weight_decimal = max_weight / 100.0

        # Section 3: Metric Weights
        st.divider()
        st.subheader("⚖️ Metric Weights")
        if is_random_mode:
            st.info(
                "🎲 **Random Mode**: Metric weights are not used for selection in random mode. "
                "Funds are selected randomly. Metrics are still calculated for reporting purposes."
            )
        st.caption("Relative importance of each metric when ranking funds for selection.")

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
                    help=HELP_TEXT.get(help_key, "Weight for this metric in fund ranking."),
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
                        value=float(model_state.get("metric_weights", {}).get(code, 1.0)),
                        step=0.1,
                        help=HELP_TEXT.get(help_key, "Weight for this metric in fund ranking."),
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
        saved_info_ratio_weight = model_state.get("metric_weights", {}).get("info_ratio", 0)
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
            rf_override_enabled = st.checkbox(
                "Override Risk-Free Rate",
                value=bool(model_state.get("rf_override_enabled", False)),
                help=HELP_TEXT["rf_override"],
            )
            # Use checkbox value directly - it's available immediately within form
            rf_rate_pct = st.number_input(
                "Constant RF Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(model_state.get("rf_rate_annual", 0.0)) * 100,
                step=0.25,
                format="%.2f",
                help=HELP_TEXT["rf_rate"],
                disabled=not rf_override_enabled,
            )
            rf_rate_annual = rf_rate_pct / 100.0 if rf_override_enabled else 0.0
            if rf_override_enabled:
                st.caption("⚠️ Using constant RF rate instead of data column.")
            else:
                st.caption("Enable override to enter a constant RF rate.")

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

            # Always show vol settings, disabled when checkbox is off (forms don't rerun)
            va_c1, va_c2, va_c3 = st.columns(3)
            with va_c1:
                vol_window_length = st.number_input(
                    "Vol Window (periods)",
                    min_value=10,
                    max_value=252,
                    value=int(model_state.get("vol_window_length", 63)),
                    help=HELP_TEXT["vol_window_length"],
                    disabled=not vol_adj_enabled,
                )
                if vol_adj_enabled:
                    st.caption(
                        f"~{vol_window_length // 21} months"
                        if vol_window_length >= 21
                        else f"{vol_window_length} days"
                    )
                else:
                    st.caption("Enable volatility adjustment to configure")

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
                        decay_methods.index(current_decay) if current_decay in decay_methods else 0
                    ),
                    help=HELP_TEXT["vol_window_decay"],
                    disabled=not vol_adj_enabled,
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
                    disabled=(not vol_adj_enabled or vol_window_decay != "ewma"),
                )
                if vol_adj_enabled and vol_window_decay == "ewma":
                    half_life = round(-1 / (1 + 1e-9 - vol_ewma_lambda), 1)
                    st.caption(f"Half-life: ~{half_life:.0f} periods")

        # Store volatility adjustment parameters in model_state
        model_state["vol_adjust_enabled"] = vol_adj_enabled
        model_state["vol_window_length"] = vol_window_length
        model_state["vol_window_decay"] = vol_window_decay
        model_state["vol_ewma_lambda"] = vol_ewma_lambda

        # Section 5: Advanced Settings
        st.divider()
        st.subheader("⚙️ Advanced Settings")
        st.caption("Fine-tune fund addition/removal rules and transaction costs.")

        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            cooldown_periods = st.number_input(
                "Cooldown Period",
                min_value=0,
                max_value=20,
                value=int(model_state.get("cooldown_periods", 1)),
                help=HELP_TEXT["cooldown_periods"],
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
            transaction_cost_bps = st.number_input(
                "Transaction Cost (bps)",
                min_value=0,
                max_value=100,
                value=int(model_state.get("transaction_cost_bps", 0)),
                help=HELP_TEXT["transaction_cost_bps"],
            )
            if transaction_cost_bps > 0:
                st.caption(f"Each trade incurs a {transaction_cost_bps} bp cost.")

        # Section 6: Fund Holding Rules (Phase 3)
        st.divider()
        st.subheader("🔒 Fund Holding Rules")
        st.caption("Control fund tenure and portfolio churn limits.")

        hold_c1, hold_c2 = st.columns(2)
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

        # NOTE: Legacy "max_active_positions" UI control removed.
        # Portfolio sizing should be configured via:
        # - Target Funds (initial): selection_count
        # - Minimum Funds: mp_min_funds
        # - Maximum Funds: mp_max_funds
        max_active_positions = 0

        # Section 7: Trend Signal Settings - REMOVED FROM UI
        # These settings require daily returns to be meaningful.
        # With monthly returns, they would be inappropriate.
        # Using default values; see docs/TrendSignalSettings.md for documentation.
        trend_window = int(model_state.get("trend_window", 63))
        trend_lag = int(model_state.get("trend_lag", 1))
        trend_min_periods_out = model_state.get("trend_min_periods")
        trend_zscore = bool(model_state.get("trend_zscore", False))
        trend_vol_adjust = bool(model_state.get("trend_vol_adjust", False))
        trend_vol_target_out = model_state.get("trend_vol_target")

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
                    c for c in benchmark_options if c.upper() not in ["SPX", "TSX", "MSCI", "ACWI"]
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

            # Reproducibility
            st.markdown("**Reproducibility**")
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
            st.caption(
                "Built-in weighting schemes are long-only unless configured to allow "
                "shorts (e.g., robust_mv min_weight < 0). This setting also affects "
                "custom weights or plugin engines configured outside the UI."
            )
            if not long_only:
                st.warning(
                    "⚠️ Short positions enabled. Ensure your data and strategy "
                    "support short selling."
                )

        # =====================================================================
        # Section 10: Entry/Exit Rules (Phase 5) - conditional on selection mode
        # =====================================================================
        st.divider()
        with st.expander("🚪 Entry/Exit Rules", expanded=False):
            # Use model_state for disabled since form checkbox changes don't apply until save
            mp_enabled_state = bool(model_state.get("multi_period_enabled", True))

            if not mp_enabled_state:
                st.warning(
                    "⚠️ Entry/exit rules only apply in multi-period mode. "
                    "Enable multi-period above, save config, then configure these settings."
                )

            st.caption(
                "Configure how funds are added to and removed from the portfolio. "
                "These settings control manager hiring and firing decisions."
            )

            # RANKING STABILITY - applies to ranking modes (Top-N, Top-%)
            if is_ranking_mode:
                st.markdown("**Ranking Stability (for Top-N / Top-% modes)**")
                st.info(
                    "Stability settings prevent churning by requiring consistent ranking "
                    "before adding or removing a fund."
                )

                rank_c1, rank_c2 = st.columns(2)
                with rank_c1:
                    sticky_add_periods = st.number_input(
                        "Periods in Top-K Before Entry",
                        min_value=1,
                        max_value=12,
                        value=int(model_state.get("sticky_add_periods", 1)),
                        help=HELP_TEXT["sticky_add_periods"],
                        disabled=not mp_enabled_state,
                    )
                    st.caption(f"Fund must rank in top-K for {sticky_add_periods} period(s).")

                with rank_c2:
                    sticky_drop_periods = st.number_input(
                        "Periods Outside Top-K Before Exit",
                        min_value=1,
                        max_value=12,
                        value=int(model_state.get("sticky_drop_periods", 1)),
                        help=HELP_TEXT["sticky_drop_periods"],
                        disabled=not mp_enabled_state,
                    )
                    st.caption(f"Fund must fall out of top-K for {sticky_drop_periods} period(s).")
            elif is_random_mode:
                # Random mode: no ranking stability needed
                st.info(
                    "🎲 **Random Mode**: Funds are randomly selected each period. "
                    "Ranking stability settings do not apply."
                )
                sticky_add_periods = 1
                sticky_drop_periods = 1
            else:
                # Defaults for threshold mode
                sticky_add_periods = 1
                sticky_drop_periods = 1

            # Z-SCORE THRESHOLDS - apply to ALL modes (important for scoring)
            st.markdown("**Z-Score Thresholds**")
            st.info(
                "Z-scores measure fund performance vs peers. Soft thresholds require "
                "consecutive periods; hard thresholds trigger immediate action."
            )

            # Soft thresholds
            st.markdown("*Soft Thresholds (consecutive periods required)*")
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
                    disabled=not mp_enabled_state,
                )
                entry_soft_strikes = st.number_input(
                    "Entry Consecutive Periods",
                    min_value=1,
                    max_value=12,
                    value=int(model_state.get("entry_soft_strikes", 1)),
                    help="Fund must pass threshold for this many consecutive periods.",
                    disabled=not mp_enabled_state,
                )
                st.caption(
                    f"Score ≥ {z_entry_soft:.2f}σ for {entry_soft_strikes} period(s) to enter."
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
                    disabled=not mp_enabled_state,
                )
                soft_strikes = st.number_input(
                    "Exit Consecutive Periods",
                    min_value=1,
                    max_value=12,
                    value=int(model_state.get("soft_strikes", 2)),
                    help="Fund must fail threshold for this many consecutive periods.",
                    disabled=not mp_enabled_state,
                )
                st.caption(f"Score ≤ {z_exit_soft:.2f}σ for {soft_strikes} period(s) to exit.")

            st.markdown("**Underweight Exit (Weight-based)**")
            min_weight_strikes = st.number_input(
                "Periods Underweight Before Forced Exit",
                min_value=0,
                max_value=12,
                value=int(model_state.get("min_weight_strikes", 2) or 2),
                help=HELP_TEXT["min_weight_strikes"],
                disabled=not mp_enabled_state,
            )
            st.caption(
                (
                    f"Same rule as engine log reason=low_weight_strikes; triggers after {min_weight_strikes} period(s)."
                    if min_weight_strikes > 0
                    else "Underweight exit is disabled."
                )
            )

            # Hard thresholds
            st.markdown("*Hard Thresholds (immediate action)*")
            hard_c1, hard_c2 = st.columns(2)
            with hard_c1:
                z_entry_hard_val = model_state.get("z_entry_hard")
                z_entry_hard_enabled = st.checkbox(
                    "Enable Hard Entry",
                    value=z_entry_hard_val is not None,
                    disabled=not mp_enabled_state,
                )
                z_entry_hard = st.number_input(
                    "Hard Entry Z-Score",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(z_entry_hard_val if z_entry_hard_val is not None else 2.0),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_entry_hard"],
                    disabled=not (mp_enabled_state and z_entry_hard_enabled),
                )
                if not z_entry_hard_enabled:
                    z_entry_hard = None
                st.caption(
                    f"Score ≥ {z_entry_hard or 2.0:.2f}σ enters instantly."
                    if z_entry_hard_enabled
                    else "Hard entry disabled."
                )

            with hard_c2:
                z_exit_hard_val = model_state.get("z_exit_hard")
                z_exit_hard_enabled = st.checkbox(
                    "Enable Hard Exit",
                    value=z_exit_hard_val is not None,
                    disabled=not mp_enabled_state,
                )
                z_exit_hard = st.number_input(
                    "Hard Exit Z-Score",
                    min_value=-5.0,
                    max_value=0.0,
                    value=float(z_exit_hard_val if z_exit_hard_val is not None else -2.0),
                    step=0.25,
                    format="%.2f",
                    help=HELP_TEXT["z_exit_hard"],
                    disabled=not (mp_enabled_state and z_exit_hard_enabled),
                )
                if not z_exit_hard_enabled:
                    z_exit_hard = None
                st.caption(
                    f"Score ≤ {z_exit_hard or -2.0:.2f}σ exits instantly."
                    if z_exit_hard_enabled
                    else "Hard exit disabled."
                )

            # Confidence Interval Reporting
            st.markdown("**Confidence Interval (Reporting Only)**")
            ci_level = st.slider(
                "Confidence Interval Level",
                min_value=0.0,
                max_value=0.99,
                value=float(model_state.get("ci_level", 0.0)),
                step=0.05,
                format="%.2f",
                help=HELP_TEXT["ci_level"],
                disabled=not mp_enabled_state,
            )
            if ci_level > 0:
                st.caption(
                    f"Reporting uses {ci_level * 100:.0f}% confidence; portfolio construction is unchanged."
                )
            else:
                st.caption("Confidence interval reporting is disabled.")

        # =====================================================================
        # Section 11: Advanced Selection Settings
        # =====================================================================
        st.divider()
        with st.expander("⚙️ Advanced Selection Settings", expanded=False):
            st.caption("Additional selection parameters for specific modes.")

            # Additional parameters for specific selection modes
            if inclusion_approach == "top_pct":
                # Use value from primary input (defined above) - convert from percentage to decimal
                rank_pct = rank_pct_input / 100.0
                st.info(
                    f"📊 Top Percentage is set to **{rank_pct_input}%** above. "
                    "Adjust it in the Fund Selection Approach section."
                )
            else:
                rank_pct = float(model_state.get("rank_pct", 0.10))

            if inclusion_approach == "threshold":
                st.info(
                    "💡 Z-Score Entry Threshold is configured in the **Entry/Exit Rules** "
                    "section below. The Entry Threshold (Z-Score) setting controls "
                    "which funds are selected."
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
                    st.caption(f"Bottom {bottom_k} ranked funds will always be excluded.")

        # =====================================================================
        # Reporting Options
        # =====================================================================
        st.markdown("---")
        with st.expander("📊 Reporting Options", expanded=False):
            st.markdown("Configure what additional information to include in the Results page.")

            report_c1, report_c2 = st.columns(2)
            with report_c1:
                show_regime_analysis = st.checkbox(
                    "Regime Performance Breakdown",
                    value=bool(model_state.get("report_regime_analysis", False)),
                    help="Show portfolio performance across different market regimes.",
                )
                show_concentration = st.checkbox(
                    "Concentration Metrics",
                    value=bool(model_state.get("report_concentration", True)),
                    help="Show HHI and effective N for portfolio concentration.",
                )
                show_benchmark_comparison = st.checkbox(
                    "Benchmark Comparison Table",
                    value=bool(model_state.get("report_benchmark_comparison", True)),
                    help="Side-by-side comparison with selected benchmarks.",
                )

            with report_c2:
                show_factor_exposures = st.checkbox(
                    "Factor Exposures",
                    value=bool(model_state.get("report_factor_exposures", False)),
                    help="Show factor exposure analysis (requires factor data).",
                )
                show_attribution = st.checkbox(
                    "Volatility-Adjusted Attribution",
                    value=bool(model_state.get("report_attribution", False)),
                    help="Contribution to return by fund, adjusted for volatility.",
                )
                show_rolling_metrics = st.checkbox(
                    "Rolling Performance Metrics",
                    value=bool(model_state.get("report_rolling_metrics", True)),
                    help="Show rolling Sharpe, IR, and other metrics over time.",
                )

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")

        if submitted:
            # Always set to Custom unless user explicitly selects Custom
            effective_preset = "Custom"

            candidate_state = {
                "preset": effective_preset,
                "lookback_periods": lookback,
                "min_history_periods": min_history,
                "evaluation_periods": evaluation,
                "multi_period_frequency": multi_period_frequency,
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
                "rf_override_enabled": rf_override_enabled,
                "rf_rate_annual": rf_rate_annual,
                "vol_floor": vol_floor,
                "warmup_periods": warmup_periods,
                "vol_adjust_enabled": vol_adj_enabled,
                "vol_window_length": vol_window_length,
                "vol_window_decay": vol_window_decay,
                "vol_ewma_lambda": vol_ewma_lambda,
                # Advanced settings
                "max_weight": max_weight_decimal,
                "min_weight": min_weight_decimal,
                "cooldown_periods": cooldown_periods,
                "rebalance_freq": rebalance_freq,
                "max_turnover": max_turnover,
                "transaction_cost_bps": transaction_cost_bps,
                # Fund holding rules (Phase 3)
                "min_tenure_periods": min_tenure_periods,
                "max_changes_per_period": max_changes_per_period,
                "max_active_positions": max_active_positions,
                # Portfolio signal parameters (Phase 4)
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
                "random_seed": random_seed,
                # Robustness fallbacks (Phase 14)
                "condition_threshold": condition_threshold,
                "safe_mode": safe_mode,
                # Constraints (Phase 15)
                "long_only": long_only,
                # Entry/Exit thresholds (Phase 5)
                "z_entry_soft": z_entry_soft,
                "z_exit_soft": z_exit_soft,
                "soft_strikes": soft_strikes,
                "entry_soft_strikes": entry_soft_strikes,
                "min_weight_strikes": min_weight_strikes,
                "sticky_add_periods": sticky_add_periods,
                "sticky_drop_periods": sticky_drop_periods,
                "ci_level": ci_level,
                # Multi-period & Selection settings (Phase 8)
                "multi_period_enabled": multi_period_enabled,
                "inclusion_approach": inclusion_approach,
                "buy_hold_initial": buy_hold_initial,
                "slippage_bps": slippage_bps,
                "bottom_k": bottom_k,
                # Selection approach details (Phase 9)
                "rank_pct": rank_pct,
                # Multi-period bounds (Phase 12)
                "mp_min_funds": mp_min_funds,
                "mp_max_funds": mp_max_funds,
                # Hard thresholds (Phase 13)
                "z_entry_hard": z_entry_hard,
                "z_exit_hard": z_exit_hard,
                # Reporting options
                "report_regime_analysis": show_regime_analysis,
                "report_concentration": show_concentration,
                "report_benchmark_comparison": show_benchmark_comparison,
                "report_factor_exposures": show_factor_exposures,
                "report_attribution": show_attribution,
                "report_rolling_metrics": show_rolling_metrics,
            }
            errors = _validate_model(candidate_state, len(fund_cols) if fund_cols else 0)
            if errors:
                st.error("\n".join(f"• {err}" for err in errors))
            else:
                st.session_state["model_state"] = candidate_state
                analysis_runner.clear_cached_analysis()
                app_state.clear_analysis_results()
                st.success("✅ Model configuration saved. Go to Results to run analysis.")


render_model_page()
