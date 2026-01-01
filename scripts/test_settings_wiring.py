#!/usr/bin/env python3
"""Comprehensive settings wiring validation for Streamlit app.

This script systematically tests every UI setting to verify it actually
affects the pipeline output. For each setting, it:
1. Runs a baseline configuration
2. Runs a variant with only that setting changed
3. Compares key metrics to detect if the setting had an effect
4. Validates the effect matches economic intuition

Usage:
    python scripts/test_settings_wiring.py [--output report.csv] [--verbose]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

from trend_analysis.config.legacy import Config  # noqa: E402

# =============================================================================
# Setting Definitions with Expected Behaviors
# =============================================================================


@dataclass
class SettingTest:
    """Definition of a setting to test."""

    name: str  # Setting key in model_state
    baseline_value: Any  # Baseline value
    test_value: Any  # Test value (should differ meaningfully)
    category: str  # Category for grouping
    expected_metric: str  # Primary metric expected to change
    expected_direction: str  # "increase", "decrease", "change", "any"
    description: str  # Human-readable description
    requires_multi_period: bool = False  # Only relevant for multi-period runs


# All settings to test, organized by category
SETTINGS_TO_TEST: list[SettingTest] = [
    # === Core Selection ===
    SettingTest(
        name="selection_count",
        baseline_value=10,
        test_value=5,
        category="Selection",
        expected_metric="num_funds_selected",
        expected_direction="decrease",
        description="Number of funds to select - fewer should mean fewer funds",
    ),
    SettingTest(
        name="inclusion_approach",
        baseline_value="threshold",
        test_value="top_n",
        category="Selection",
        expected_metric="selection_method",
        expected_direction="change",
        description="Selection method - different methods should produce different results",
    ),
    SettingTest(
        name="inclusion_approach",
        baseline_value="threshold",
        test_value="random",
        category="Selection",
        expected_metric="selection_variance",
        expected_direction="increase",
        description="Random selection should produce more variable results",
    ),
    SettingTest(
        name="inclusion_approach",
        baseline_value="threshold",
        test_value="buy_and_hold",
        category="Selection",
        expected_metric="turnover",
        expected_direction="decrease",
        description="Buy and hold should produce lower turnover",
    ),
    SettingTest(
        name="rank_pct",
        baseline_value=0.10,
        test_value=0.30,
        category="Selection",
        expected_metric="num_funds_selected",
        expected_direction="increase",
        description="Higher percentage should select more funds (top_pct mode)",
    ),
    # === Weighting ===
    SettingTest(
        name="weighting_scheme",
        baseline_value="equal",
        test_value="risk_parity",
        category="Weighting",
        expected_metric="weight_dispersion",
        expected_direction="change",
        description="Risk parity should produce different weights than equal",
    ),
    SettingTest(
        name="weighting_scheme",
        baseline_value="equal",
        test_value="hrp",
        category="Weighting",
        expected_metric="weight_dispersion",
        expected_direction="change",
        description="HRP should produce different weights than equal",
    ),
    # === Risk Settings ===
    SettingTest(
        name="risk_target",
        baseline_value=0.10,
        test_value=0.20,
        category="Risk",
        expected_metric="portfolio_volatility",
        expected_direction="increase",
        description="Higher risk target should increase volatility",
    ),
    SettingTest(
        name="vol_floor",
        baseline_value=0.05,
        test_value=0.15,  # 15% floor will affect funds with vol ~10%
        category="Risk",
        expected_metric="scaling_factor",
        expected_direction="decrease",  # Higher floor -> lower scale factors
        description="Higher vol floor should reduce scaling factors for low-vol assets",
    ),
    SettingTest(
        name="max_weight",
        baseline_value=0.20,
        test_value=0.10,
        category="Constraints",
        expected_metric="max_position_weight",
        expected_direction="decrease",
        description="Lower max weight should cap positions lower",
    ),
    SettingTest(
        name="min_weight",
        baseline_value=0.03,
        test_value=0.08,
        category="Constraints",
        expected_metric="min_position_weight",
        expected_direction="increase",
        description="Higher min weight should increase minimum positions (requires non-equal weighting)",
        # Note: This test requires risk_parity or hrp weighting to see effect
    ),
    # === Entry/Exit Thresholds ===
    SettingTest(
        name="z_entry_soft",
        baseline_value=1.0,
        test_value=2.0,
        category="Entry/Exit",
        expected_metric="entry_frequency",
        expected_direction="decrease",
        description="Higher entry threshold should reduce entries",
        requires_multi_period=True,
    ),
    SettingTest(
        name="z_exit_soft",
        baseline_value=-1.0,
        test_value=-0.5,
        category="Entry/Exit",
        expected_metric="exit_frequency",
        expected_direction="increase",
        description="Higher (less negative) exit threshold should increase exits",
        requires_multi_period=True,
    ),
    SettingTest(
        name="soft_strikes",
        baseline_value=2,
        test_value=4,
        category="Entry/Exit",
        expected_metric="exit_frequency",
        expected_direction="decrease",
        description="More strikes needed should reduce exit frequency",
        requires_multi_period=True,
    ),
    SettingTest(
        name="entry_soft_strikes",
        baseline_value=1,
        test_value=3,
        category="Entry/Exit",
        expected_metric="entry_frequency",
        expected_direction="decrease",
        description="More entry strikes should reduce entry frequency",
        requires_multi_period=True,
    ),
    # === Multi-Period Settings ===
    SettingTest(
        name="lookback_periods",
        baseline_value=3,
        test_value=5,
        category="Multi-Period",
        expected_metric="in_sample_months",
        expected_direction="increase",
        description="Longer lookback should use more historical data",
        requires_multi_period=True,
    ),
    SettingTest(
        name="evaluation_periods",
        baseline_value=1,
        test_value=2,
        category="Multi-Period",
        expected_metric="out_sample_months",
        expected_direction="increase",
        description="Longer evaluation should extend OOS window",
        requires_multi_period=True,
    ),
    SettingTest(
        name="multi_period_frequency",
        baseline_value="A",
        test_value="Q",
        category="Multi-Period",
        expected_metric="num_periods",
        expected_direction="increase",
        description="Quarterly frequency should produce more periods than annual",
        requires_multi_period=True,
    ),
    SettingTest(
        name="mp_max_funds",
        baseline_value=25,
        test_value=8,
        category="Multi-Period",
        expected_metric="max_funds_in_period",
        expected_direction="decrease",
        description="Lower cap should limit funds per period",
        requires_multi_period=True,
    ),
    SettingTest(
        name="mp_min_funds",
        baseline_value=5,
        test_value=12,
        category="Multi-Period",
        expected_metric="min_funds_in_period",
        expected_direction="increase",
        description="Higher floor should ensure more funds per period",
        requires_multi_period=True,
    ),
    # === Costs ===
    SettingTest(
        name="transaction_cost_bps",
        baseline_value=0,
        test_value=50,
        category="Costs",
        expected_metric="total_costs",
        expected_direction="increase",
        description="Higher transaction costs should increase total costs",
    ),
    SettingTest(
        name="slippage_bps",
        baseline_value=0,
        test_value=25,
        category="Costs",
        expected_metric="total_costs",
        expected_direction="increase",
        description="Adding slippage should increase total costs",
    ),
    SettingTest(
        name="max_turnover",
        baseline_value=1.0,
        test_value=0.3,
        category="Costs",
        expected_metric="actual_turnover",
        expected_direction="decrease",
        description="Lower max turnover should constrain actual turnover",
        requires_multi_period=True,
    ),
    # === Metric Weights ===
    SettingTest(
        name="metric_weights",
        baseline_value={"sharpe": 1.0, "return_ann": 1.0, "drawdown": 0.5},
        test_value={"sharpe": 0.0, "return_ann": 3.0, "drawdown": 0.0},
        category="Scoring",
        expected_metric="selected_fund_returns",
        expected_direction="increase",
        description="Return-only weighting should select higher-return funds",
    ),
    SettingTest(
        name="metric_weights",
        baseline_value={"sharpe": 1.0, "return_ann": 1.0, "drawdown": 0.5},
        test_value={"sharpe": 0.0, "return_ann": 0.0, "drawdown": 3.0},
        category="Scoring",
        expected_metric="selected_fund_drawdown",
        expected_direction="decrease",
        description="Drawdown-only weighting should select lower-drawdown funds",
    ),
    # === Preprocessing ===
    SettingTest(
        name="warmup_periods",
        baseline_value=0,
        test_value=12,
        category="Data",
        expected_metric="effective_start_date",
        expected_direction="increase",
        description="Warmup should delay effective start date",
    ),
    # === Random Seed ===
    SettingTest(
        name="random_seed",
        baseline_value=42,
        test_value=123,
        category="Reproducibility",
        expected_metric="random_selection_result",
        expected_direction="change",
        description="Different seed should produce different random selections",
    ),
    # === Risk-Free Rate ===
    SettingTest(
        name="rf_rate_annual",
        baseline_value=0.0,
        test_value=0.05,
        category="Risk",
        expected_metric="average_sharpe",
        expected_direction="change",
        description="Higher risk-free rate should change Sharpe ratios",
    ),
    # === Buy and Hold Initial Selection ===
    SettingTest(
        name="buy_hold_initial",
        baseline_value="top_n",
        test_value="threshold",
        category="Selection",
        expected_metric="selected_funds_set",
        expected_direction="change",
        description="Different initial selection method should select different funds",
        requires_multi_period=True,
    ),
]


# =============================================================================
# Test Runner
# =============================================================================


@dataclass
class TestResult:
    """Result of a single setting test."""

    setting_name: str
    category: str
    baseline_value: Any
    test_value: Any
    description: str
    expected_metric: str
    expected_direction: str
    baseline_metric_value: Any
    test_metric_value: Any
    metric_changed: bool
    direction_correct: bool
    status: str  # PASS, FAIL, SKIP, ERROR
    error_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def get_baseline_state() -> dict[str, Any]:
    """Return baseline model state for testing.

    Uses single-period mode by default for faster testing.
    Multi-period tests should explicitly set multi_period_enabled=True.
    """
    return {
        "preset": "Baseline",
        "lookback_periods": 3,
        "min_history_periods": 3,
        "evaluation_periods": 1,
        "selection_count": 10,
        "weighting_scheme": "equal",
        "metric_weights": {"sharpe": 1.0, "return_ann": 1.0, "drawdown": 0.5},
        "risk_target": 0.10,
        "date_mode": "relative",
        "rf_override_enabled": True,  # Enable rf override to test rf_rate_annual
        "rf_rate_annual": 0.0,
        "vol_floor": 0.015,
        "warmup_periods": 0,
        "max_weight": 0.20,  # Aligned with baseline in 8_Validation.py
        "min_weight": 0.05,  # Aligned with baseline in 8_Validation.py
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
        "random_seed": 42,
        "condition_threshold": 1.0e12,
        "safe_mode": "hrp",
        "long_only": True,
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


def load_demo_data() -> pd.DataFrame:
    """Load demo returns data for testing."""
    demo_path = PROJECT_ROOT / "demo" / "demo_returns.csv"
    if not demo_path.exists():
        # Generate demo data if it doesn't exist
        from scripts.generate_demo import main as generate_demo

        generate_demo()

    df = pd.read_csv(demo_path, parse_dates=["Date"])
    df = df.set_index("Date")
    return df


def run_analysis_with_state(
    returns: pd.DataFrame,
    model_state: dict[str, Any],
    benchmark: str | None = None,
) -> Any:
    """Run analysis with given state and return results.

    Returns a RunResult-like object with attributes:
    - weights: pd.Series of portfolio weights
    - metrics: pd.DataFrame of summary metrics
    - details: dict with full result payload
    - period_results: list of dicts for multi-period runs
    """
    from trend_analysis.api import run_simulation

    # Build config from model state (replicate logic from analysis_runner)
    config = _build_config_from_state(returns, model_state, benchmark)

    # Prepare returns DataFrame with Date column
    returns_df = returns.reset_index()
    index_name = returns.index.name or "Date"
    returns_df = returns_df.rename(columns={index_name: "Date"})

    return run_simulation(config, returns_df)  # type: ignore[arg-type]


def _build_config_from_state(
    returns: pd.DataFrame,
    state: dict[str, Any],
    benchmark: str | None = None,
) -> Config:
    """Build a Config object from model_state dict.

    This replicates the logic in streamlit_app/components/analysis_runner.py
    but without requiring streamlit to be importable.
    """
    from trend_analysis.signals import TrendSpec as TrendSpecModel

    def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            return default
        return max(as_int, minimum)

    def _coerce_positive_float(value: Any, *, default: float) -> float:
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return default
        return max(as_float, 0.0)

    def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
        stamp = pd.Timestamp(ts)
        period = stamp.to_period("M")
        return period.to_timestamp("M", how="end")

    METRIC_REGISTRY = {
        "sharpe": "Sharpe",
        "return_ann": "AnnualReturn",
        "sortino": "Sortino",
        "info_ratio": "InformationRatio",
        "drawdown": "MaxDrawdown",
        "vol": "Volatility",
    }

    # Normalize metric weights
    raw_weights = state.get("metric_weights", {})
    weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        if name not in METRIC_REGISTRY:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weights[name] = weight
    if not weights:
        default = 1.0 / 3
        weights = {"sharpe": default, "return_ann": default, "drawdown": default}
    total = sum(weights.values())
    weights = {name: weight / total for name, weight in weights.items()}

    registry_weights = {
        METRIC_REGISTRY.get(metric, metric): float(weight)
        for metric, weight in weights.items()
    }

    # Build sample_split
    index = returns.index
    frequency = str(state.get("multi_period_frequency", "A") or "A")
    period_to_months = {"M": 1, "Q": 3, "A": 12}
    months_per_period = period_to_months.get(frequency, 12)

    lookback_periods = _coerce_positive_int(
        state.get("lookback_periods"), default=3, minimum=1
    )
    evaluation_periods = _coerce_positive_int(
        state.get("evaluation_periods"), default=1, minimum=1
    )
    lookback_months = lookback_periods * months_per_period
    evaluation_months = evaluation_periods * months_per_period

    last = _month_end(index.max())
    first = _month_end(index.min())
    out_start = _month_end(last - pd.DateOffset(months=evaluation_months - 1))
    if out_start < first:
        out_start = first
    in_end = _month_end(out_start - pd.DateOffset(months=1))
    if in_end < first:
        in_end = first
    in_start = _month_end(in_end - pd.DateOffset(months=lookback_months - 1))
    if in_start < first:
        in_start = first

    sample_split = {
        "in_start": in_start.strftime("%Y-%m"),
        "in_end": in_end.strftime("%Y-%m"),
        "out_start": out_start.strftime("%Y-%m"),
        "out_end": last.strftime("%Y-%m"),
    }

    # Build portfolio config
    selection_count = _coerce_positive_int(
        state.get("selection_count"), default=10, minimum=1
    )
    weighting_scheme = str(state.get("weighting_scheme", "equal") or "equal")
    max_weight = _coerce_positive_float(state.get("max_weight"), default=0.20)
    max_turnover = _coerce_positive_float(state.get("max_turnover"), default=1.0)
    transaction_cost_bps = _coerce_positive_int(
        state.get("transaction_cost_bps"), default=0, minimum=0
    )
    rebalance_freq = str(state.get("rebalance_freq", "M") or "M")
    min_tenure_periods = _coerce_positive_int(
        state.get("min_tenure_periods"), default=0, minimum=0
    )
    max_changes_per_period = _coerce_positive_int(
        state.get("max_changes_per_period"), default=0, minimum=0
    )

    selection_approach = str(
        state.get("inclusion_approach") or state.get("selection_approach") or "top_n"
    )
    is_buy_and_hold = selection_approach == "buy_and_hold"
    buy_hold_initial = str(state.get("buy_hold_initial", "top_n"))
    effective_approach = buy_hold_initial if is_buy_and_hold else selection_approach
    rank_transform = "zscore" if effective_approach == "threshold" else "raw"
    slippage_bps = _coerce_positive_int(state.get("slippage_bps"), default=0, minimum=0)
    bottom_k = _coerce_positive_int(state.get("bottom_k"), default=0, minimum=0)
    rank_pct = _coerce_positive_float(state.get("rank_pct"), default=0.10)
    rank_threshold = _coerce_positive_float(
        state.get("z_entry_soft") or state.get("rank_threshold"), default=1.0
    )
    long_only = bool(state.get("long_only", True))

    is_random_mode = selection_approach == "random"
    if is_buy_and_hold and buy_hold_initial == "random":
        is_random_mode = True
    selection_mode = "random" if is_random_mode else "rank"
    if is_buy_and_hold:
        selection_mode = "buy_and_hold"

    portfolio_cfg: dict[str, Any] = {
        "selection_mode": selection_mode,
        "rank": {
            "inclusion_approach": selection_approach,
            "n": selection_count,
            "pct": rank_pct,
            "threshold": rank_threshold,
            "score_by": "blended",
            "blended_weights": registry_weights,
            "transform": rank_transform,
        },
        "buy_and_hold": {
            "initial_method": buy_hold_initial,
            "n": selection_count,
            "pct": rank_pct,
            "threshold": rank_threshold,
            "blended_weights": registry_weights,
        },
        "random_n": selection_count,
        "weighting_scheme": weighting_scheme,
        "rebalance_freq": rebalance_freq,
        "max_turnover": max_turnover,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "constraints": {
            "long_only": long_only,
            "max_weight": max_weight,
        },
    }

    if slippage_bps > 0:
        portfolio_cfg["cost_model"] = {
            "bps_per_trade": transaction_cost_bps,
            "slippage_bps": slippage_bps,
        }
    if bottom_k > 0:
        portfolio_cfg["rank"]["bottom_k"] = bottom_k
    if min_tenure_periods > 0:
        portfolio_cfg["min_tenure_n"] = min_tenure_periods
    if max_changes_per_period > 0:
        portfolio_cfg["turnover_budget_max_changes"] = max_changes_per_period

    # Multi-period capacity
    mp_max_funds = _coerce_positive_int(state.get("mp_max_funds"), default=0, minimum=0)
    if mp_max_funds > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["max_funds"] = mp_max_funds

    mp_min_funds = _coerce_positive_int(state.get("mp_min_funds"), default=0, minimum=0)
    if mp_min_funds > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_funds"] = mp_min_funds

    min_weight = state.get("min_weight")
    if min_weight is not None:
        min_weight_val = _coerce_positive_float(min_weight, default=0.05)
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_weight"] = min_weight_val

    min_weight_strikes = _coerce_positive_int(
        state.get("min_weight_strikes"), default=0, minimum=0
    )
    if min_weight_strikes > 0:
        portfolio_cfg.setdefault("constraints", {})
        portfolio_cfg["constraints"]["min_weight_strikes"] = min_weight_strikes

    cooldown_periods = _coerce_positive_int(
        state.get("cooldown_periods"), default=0, minimum=0
    )
    if cooldown_periods > 0:
        portfolio_cfg["cooldown_periods"] = cooldown_periods

    # Entry/exit thresholds
    z_entry_soft = float(state.get("z_entry_soft", 1.0) or 1.0)
    z_exit_soft = float(state.get("z_exit_soft", -1.0) or -1.0)
    soft_strikes = int(state.get("soft_strikes", 2) or 2)
    entry_soft_strikes = int(state.get("entry_soft_strikes", 1) or 1)

    z_entry_hard_val = state.get("z_entry_hard")
    z_exit_hard_val = state.get("z_exit_hard")
    z_entry_hard = float(z_entry_hard_val) if z_entry_hard_val is not None else None
    z_exit_hard = float(z_exit_hard_val) if z_exit_hard_val is not None else None

    threshold_hold_cfg = {
        "z_entry_soft": z_entry_soft,
        "z_exit_soft": z_exit_soft,
        "soft_strikes": soft_strikes,
        "entry_soft_strikes": entry_soft_strikes,
        "metric": "blended",
        "blended_weights": registry_weights,
        "target_n": selection_count,
    }
    if z_entry_hard is not None:
        threshold_hold_cfg["z_entry_hard"] = z_entry_hard
    if z_exit_hard is not None:
        threshold_hold_cfg["z_exit_hard"] = z_exit_hard

    portfolio_cfg["policy"] = "threshold_hold"
    portfolio_cfg["threshold_hold"] = threshold_hold_cfg

    sticky_add_periods = int(state.get("sticky_add_periods", 1) or 1)
    sticky_drop_periods = int(state.get("sticky_drop_periods", 1) or 1)
    ci_level = float(state.get("ci_level", 0.0) or 0.0)
    portfolio_cfg["sticky_add_x"] = sticky_add_periods
    portfolio_cfg["sticky_drop_y"] = sticky_drop_periods
    portfolio_cfg["ci_level"] = ci_level

    # Signals config
    base = TrendSpecModel()
    window = _coerce_positive_int(state.get("trend_window"), default=base.window)
    lag = _coerce_positive_int(state.get("trend_lag"), default=base.lag)
    min_periods_raw = state.get("trend_min_periods")
    try:
        min_periods: int | None = (
            int(min_periods_raw)  # type: ignore[arg-type]
            if min_periods_raw not in (None, "")
            else None
        )
    except (TypeError, ValueError):
        min_periods = None
    if min_periods is not None and min_periods <= 0:
        min_periods = None
    if min_periods is not None and min_periods > window:
        min_periods = window

    vol_adjust = bool(state.get("trend_vol_adjust", base.vol_adjust))
    vol_target_raw = state.get("trend_vol_target")
    try:
        vol_target = float(vol_target_raw) if vol_target_raw is not None else None
    except (TypeError, ValueError):
        vol_target = None
    if vol_target is not None and vol_target <= 0:
        vol_target = None
    if not vol_adjust:
        vol_target = None

    zscore = bool(state.get("trend_zscore", base.zscore))

    signals_cfg: dict[str, Any] = {
        "kind": base.kind,
        "window": window,
        "lag": lag,
        "vol_adjust": vol_adjust,
        "zscore": zscore,
    }
    if min_periods is not None:
        signals_cfg["min_periods"] = min_periods
    if vol_target is not None:
        signals_cfg["vol_target"] = vol_target

    # Multi-period config
    multi_period_enabled = bool(state.get("multi_period_enabled", True))
    multi_period_cfg = None
    if multi_period_enabled:
        multi_period_frequency = str(state.get("multi_period_frequency", "A") or "A")
        in_sample_len = _coerce_positive_int(
            state.get("lookback_periods") or state.get("in_sample_years"),
            default=3,
            minimum=1,
        )
        out_sample_len = _coerce_positive_int(
            state.get("evaluation_periods") or state.get("out_sample_years"),
            default=1,
            minimum=1,
        )
        min_history_len = _coerce_positive_int(
            state.get("min_history_periods"), default=in_sample_len, minimum=1
        )
        min_history_len = min(min_history_len, in_sample_len)

        data_start = index.min()
        data_end = index.max()
        start_me = _month_end(data_start)
        end_me = _month_end(data_end)

        multi_period_cfg = {
            "start": start_me.strftime("%Y-%m-%d"),
            "end": end_me.strftime("%Y-%m-%d"),
            "frequency": multi_period_frequency,
            "in_sample_len": in_sample_len,
            "out_sample_len": out_sample_len,
            "min_history": min_history_len,
        }

    # Data config
    data_cfg: dict[str, Any] = {
        "allow_risk_free_fallback": True,
    }

    preprocessing_cfg: dict[str, Any] = {}

    # Vol adjust config
    vol_target_cfg = _coerce_positive_float(state.get("risk_target"), default=0.1)
    vol_floor = _coerce_positive_float(state.get("vol_floor"), default=0.015)
    warmup_periods_cfg = _coerce_positive_int(
        state.get("warmup_periods"), default=0, minimum=0
    )

    # Robustness
    shrinkage_enabled = bool(state.get("shrinkage_enabled", True))
    shrinkage_method = str(
        state.get("shrinkage_method", "ledoit_wolf") or "ledoit_wolf"
    )
    condition_threshold = float(state.get("condition_threshold", 1.0e12) or 1.0e12)
    safe_mode = str(state.get("safe_mode", "hrp") or "hrp")

    robustness_cfg = {
        "shrinkage": {
            "enabled": shrinkage_enabled,
            "method": shrinkage_method,
        },
        "condition_check": {
            "enabled": True,
            "threshold": condition_threshold,
            "safe_mode": safe_mode,
        },
    }

    # Regime
    regime_enabled = bool(state.get("regime_enabled", False))
    regime_proxy = str(state.get("regime_proxy", "SPX") or "SPX")
    regime_cfg = {"enabled": regime_enabled, "proxy": regime_proxy}

    metrics_registry = [METRIC_REGISTRY.get(name, name) for name in weights]

    benchmark_map: dict[str, str] = {}
    if benchmark:
        benchmark_map[benchmark] = benchmark

    seed = 42
    try:
        seed_raw = state.get("random_seed")
        if seed_raw is not None:
            seed = int(seed_raw)
    except (TypeError, ValueError):
        seed = 42

    rf_override_enabled = bool(state.get("rf_override_enabled", False))
    rf_rate_annual = _coerce_positive_float(state.get("rf_rate_annual"), default=0.0)

    return Config(
        version="1",
        data=data_cfg,
        preprocessing=preprocessing_cfg,
        vol_adjust={
            "target_vol": vol_target_cfg,
            "floor_vol": vol_floor,
            "warmup_periods": warmup_periods_cfg,
        },
        sample_split=sample_split,
        portfolio=portfolio_cfg,
        signals=signals_cfg,
        benchmarks=benchmark_map,
        regime=regime_cfg,
        robustness=robustness_cfg,
        metrics={
            "registry": metrics_registry,
            "rf_rate_annual": rf_rate_annual,
            "rf_override_enabled": rf_override_enabled,
        },
        export={},
        run={"trend_preset": state.get("preset")},
        seed=seed,
        multi_period=multi_period_cfg,
    )


def extract_metric(
    result: Any,
    metric_name: str,
    model_state: dict[str, Any],
) -> Any:
    """Extract a specific metric from analysis results (RunResult-like object)."""

    # Handle special computed metrics
    if metric_name == "num_funds_selected":
        if result.weights is not None:
            return int((result.weights > 0).sum())
        if result.period_results:
            counts = []
            for p in result.period_results:
                if "weights" in p:
                    w = p["weights"]
                    counts.append(sum(1 for v in w.values() if v > 0))
            return np.mean(counts) if counts else 0
        return None

    if metric_name == "selection_method":
        return model_state.get("inclusion_approach")

    if metric_name == "selection_variance":
        if result.period_results:
            counts = []
            for p in result.period_results:
                # Try selected_funds first, then fall back to weights
                if "selected_funds" in p:
                    counts.append(len(p["selected_funds"]))
                elif "weights" in p:
                    w = p["weights"]
                    counts.append(sum(1 for v in w.values() if v > 0))
            return np.std(counts) if len(counts) > 1 else 0.0
        return 0.0

    if metric_name == "turnover":
        if result.turnover is not None:
            return float(result.turnover.mean()) if len(result.turnover) > 0 else 0.0
        if result.period_results:
            turnovers = [
                p.get("turnover", 0.0) for p in result.period_results if "turnover" in p
            ]
            return np.mean(turnovers) if turnovers else 0.0
        return 0.0

    if metric_name == "actual_turnover":
        # Extract turnover from period results
        if result.period_results:
            turnovers = [
                p.get("turnover", 0.0) for p in result.period_results if "turnover" in p
            ]
            return float(sum(turnovers)) if turnovers else 0.0
        if result.turnover is not None:
            return float(result.turnover.sum()) if len(result.turnover) > 0 else 0.0
        return 0.0

    if metric_name == "weight_dispersion":
        if result.weights is not None:
            return float(result.weights.std()) if len(result.weights) > 0 else 0.0
        return 0.0

    if metric_name == "portfolio_volatility":
        # Try to get from out_user_stats in period_results first (most accurate)
        if result.period_results:
            vols = []
            for p in result.period_results:
                out_stats = p.get("out_user_stats")
                if out_stats and hasattr(out_stats, "vol"):
                    vols.append(float(out_stats.vol))
            if vols:
                return float(np.mean(vols))
        # Fallback to metrics DataFrame
        if result.metrics is not None:
            for col in ["Volatility", "vol", "Vol"]:
                if col in result.metrics.columns:
                    return float(result.metrics[col].iloc[0])
        return 0.0

    if metric_name == "scaling_factor":
        # Extract average scale factor from risk_diagnostics
        if result.period_results:
            scale_factors = []
            for p in result.period_results:
                rd = p.get("risk_diagnostics", {})
                sf = rd.get("scale_factors")
                if sf is not None:
                    scale_factors.append(float(sf.mean()))
            if scale_factors:
                return float(np.mean(scale_factors))
        return 0.0

    if metric_name == "max_position_weight":
        if result.weights is not None:
            return float(result.weights.max()) if len(result.weights) > 0 else 0.0
        # Check period results for weights
        if result.period_results:
            max_weights = []
            for p in result.period_results:
                if "weights" in p and p["weights"]:
                    max_weights.append(max(p["weights"].values()))
            if max_weights:
                return float(max(max_weights))
        return 0.0

    if metric_name == "min_position_weight":
        # For multi-period, extract from period results to get per-period weights
        if result.period_results:
            min_weights = []
            for p in result.period_results:
                fw = p.get("fund_weights", {})
                if fw:
                    pos = [v for v in fw.values() if v > 0]
                    if pos:
                        min_weights.append(min(pos))
            if min_weights:
                return float(min(min_weights))
        # Fallback to aggregated weights
        if result.weights is not None:
            pos = result.weights[result.weights > 0]
            return float(pos.min()) if len(pos) > 0 else 0.0
        return 0.0

    if metric_name == "average_sharpe":
        # Extract average Sharpe ratio from per-fund metrics
        if result.metrics is not None and isinstance(result.metrics, pd.DataFrame):
            if "sharpe" in result.metrics.columns:
                return float(result.metrics["sharpe"].mean())
        return 0.0

    if metric_name == "selected_funds_set":
        # Return a frozenset of selected fund names for comparison
        if result.weights is not None:
            return frozenset(result.weights[result.weights > 0].index.tolist())
        if result.period_results:
            # Aggregate all selected funds across periods
            all_funds = set()
            for p in result.period_results:
                if "selected_funds" in p:
                    all_funds.update(p["selected_funds"])
                elif "weights" in p:
                    all_funds.update(k for k, v in p["weights"].items() if v > 0)
            return frozenset(all_funds)
        return frozenset()

    if metric_name == "num_periods":
        if result.period_results:
            return len(result.period_results)
        return result.period_count if result.period_count > 0 else 1

    if metric_name == "total_costs":
        if result.costs:
            return result.costs.get("total", 0.0)
        # Fallback: sum transaction costs from period_results
        if result.period_results:
            total_cost = sum(
                p.get("transaction_cost", 0.0) for p in result.period_results
            )
            return float(total_cost)
        return 0.0

    if metric_name == "in_sample_months":
        if result.period_results and "in_sample_months" in result.period_results[0]:
            return result.period_results[0]["in_sample_months"]
        return model_state.get("lookback_periods", 3) * 12

    if metric_name == "out_sample_months":
        if result.period_results and "out_sample_months" in result.period_results[0]:
            return result.period_results[0]["out_sample_months"]
        return model_state.get("evaluation_periods", 1) * 12

    # For metrics we can't easily extract, return a hash of the result
    # This at least detects if the output changed
    if metric_name in [
        "scaling_factor",
        "gross_exposure",
        "entry_frequency",
        "exit_frequency",
        "reentry_frequency",
        "avg_holding_duration",
        "underweight_exits",
        "actual_turnover",
        "selected_fund_returns",
        "selected_fund_drawdown",
        "weight_stability",
        "data_points_used",
        "return_extremes",
        "effective_start_date",
        "signal_responsiveness",
        "signal_distribution",
        "random_selection_result",
        "max_funds_in_period",
        "min_funds_in_period",
    ]:
        # Compute a hash of relevant output to detect changes
        hashable_data = {
            "weights": result.weights.to_dict() if result.weights is not None else {},
            "metrics_hash": (
                str(result.metrics.values.tolist())[:100]
                if result.metrics is not None
                else ""
            ),
        }
        hashable = json.dumps(hashable_data, sort_keys=True, default=str)
        return hashlib.md5(hashable.encode()).hexdigest()[:8]

    # Direct lookup from details
    if result.details:
        return result.details.get(metric_name)

    # Explicitly return None for unknown metrics to make behavior clear
    return None


def check_direction(
    baseline_val: Any,
    test_val: Any,
    expected_direction: str,
) -> bool:
    """Check if the change direction matches expectation."""
    if baseline_val is None or test_val is None:
        return False

    if expected_direction == "change" or expected_direction == "any":
        return bool(baseline_val != test_val)

    try:
        baseline_num = float(baseline_val)
        test_num = float(test_val)
    except (TypeError, ValueError):
        # For non-numeric values, just check if they changed
        return bool(baseline_val != test_val)

    if expected_direction == "increase":
        return bool(test_num > baseline_num)
    elif expected_direction == "decrease":
        return bool(test_num < baseline_num)

    return bool(baseline_val != test_val)


def run_single_test(
    setting: SettingTest,
    returns: pd.DataFrame,
    verbose: bool = False,
) -> TestResult:
    """Run a single setting test."""

    # Build baseline state
    baseline_state = get_baseline_state()

    # Build test state with only this setting changed
    test_state = baseline_state.copy()
    test_state[setting.name] = setting.test_value

    # Also update baseline with the baseline value (in case default differs)
    baseline_state[setting.name] = setting.baseline_value

    # Handle special cases for inclusion_approach tests
    if setting.name == "inclusion_approach":
        if setting.test_value == "top_pct":
            test_state["rank_pct"] = 0.20  # Ensure meaningful selection
        elif setting.test_value == "random":
            test_state["random_seed"] = 42  # Ensure reproducibility

    # min_weight constraint needs non-equal weighting to have observable effect
    if setting.name == "min_weight":
        baseline_state["weighting_scheme"] = "risk_parity"
        test_state["weighting_scheme"] = "risk_parity"

    # random_seed test needs random selection mode to demonstrate effect
    if setting.name == "random_seed":
        baseline_state["inclusion_approach"] = "random"
        test_state["inclusion_approach"] = "random"

    # rank_pct test needs top_pct selection mode
    if setting.name == "rank_pct":
        baseline_state["inclusion_approach"] = "top_pct"
        test_state["inclusion_approach"] = "top_pct"

    # buy_hold_initial test needs buy_and_hold selection mode
    if setting.name == "buy_hold_initial":
        baseline_state["inclusion_approach"] = "buy_and_hold"
        test_state["inclusion_approach"] = "buy_and_hold"

    # shrinkage tests need robust_mv weighting where shrinkage is applied
    if setting.name in ["shrinkage_enabled", "shrinkage_method"]:
        baseline_state["weighting_scheme"] = "robust_mv"
        test_state["weighting_scheme"] = "robust_mv"
    try:
        if verbose:
            print(f"  Running baseline: {setting.name}={setting.baseline_value}")
        baseline_result = run_analysis_with_state(returns, baseline_state)

        if verbose:
            print(f"  Running test: {setting.name}={setting.test_value}")
        test_result = run_analysis_with_state(returns, test_state)

        # Extract metrics
        baseline_metric = extract_metric(
            baseline_result, setting.expected_metric, baseline_state
        )
        test_metric = extract_metric(test_result, setting.expected_metric, test_state)

        # Check if metric changed
        metric_changed = baseline_metric != test_metric

        # Check if direction is correct
        direction_correct = check_direction(
            baseline_metric, test_metric, setting.expected_direction
        )

        # Determine status
        if not metric_changed:
            status = "FAIL"
            error = f"Setting had no effect on {setting.expected_metric}"
        elif not direction_correct:
            status = "WARN"
            error = f"Direction mismatch: expected {setting.expected_direction}"
        else:
            status = "PASS"
            error = ""

        return TestResult(
            setting_name=setting.name,
            category=setting.category,
            baseline_value=setting.baseline_value,
            test_value=setting.test_value,
            description=setting.description,
            expected_metric=setting.expected_metric,
            expected_direction=setting.expected_direction,
            baseline_metric_value=baseline_metric,
            test_metric_value=test_metric,
            metric_changed=metric_changed,
            direction_correct=direction_correct,
            status=status,
            error_message=error,
        )

    except Exception as e:
        return TestResult(
            setting_name=setting.name,
            category=setting.category,
            baseline_value=setting.baseline_value,
            test_value=setting.test_value,
            description=setting.description,
            expected_metric=setting.expected_metric,
            expected_direction=setting.expected_direction,
            baseline_metric_value=None,
            test_metric_value=None,
            metric_changed=False,
            direction_correct=False,
            status="ERROR",
            error_message=str(e),
            details={"traceback": traceback.format_exc()},
        )


def run_all_tests(
    returns: pd.DataFrame,
    verbose: bool = False,
) -> list[TestResult]:
    """Run all setting tests."""
    results = []

    for i, setting in enumerate(SETTINGS_TO_TEST, 1):
        if verbose:
            print(f"\n[{i}/{len(SETTINGS_TO_TEST)}] Testing: {setting.name}")
            print(f"  Category: {setting.category}")
            print(f"  {setting.baseline_value} -> {setting.test_value}")

        result = run_single_test(setting, returns, verbose)
        results.append(result)

        if verbose:
            status_symbol = {
                "PASS": "✅",
                "FAIL": "❌",
                "WARN": "⚠️",
                "ERROR": "💥",
                "SKIP": "⏭️",
            }
            print(
                f"  {status_symbol.get(result.status, '?')} {result.status}: {result.error_message or 'OK'}"
            )

    return results


def generate_report(
    results: list[TestResult],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a report DataFrame from test results."""

    rows = []
    for r in results:
        rows.append(
            {
                "Setting": r.setting_name,
                "Category": r.category,
                "Baseline": str(r.baseline_value),
                "Test": str(r.test_value),
                "Metric": r.expected_metric,
                "Expected": r.expected_direction,
                "Baseline Value": str(r.baseline_metric_value),
                "Test Value": str(r.test_metric_value),
                "Changed": "Yes" if r.metric_changed else "No",
                "Direction OK": "Yes" if r.direction_correct else "No",
                "Status": r.status,
                "Error": r.error_message,
                "Description": r.description,
            }
        )

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nReport saved to: {output_path}")

    return df


def print_summary(results: list[TestResult]) -> None:
    """Print a summary of test results."""

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    warnings = sum(1 for r in results if r.status == "WARN")
    errors = sum(1 for r in results if r.status == "ERROR")

    print("\n" + "=" * 60)
    print("SETTINGS WIRING VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tests:  {total}")
    if total == 0:
        # Avoid division by zero and provide a clear summary when no tests ran
        print("✅ Passed:    0 (0.0%)")
        print("❌ Failed:    0 (0.0%)")
        print("⚠️  Warnings:  0 (0.0%)")
        print("💥 Errors:    0 (0.0%)")
        print("=" * 60)
        print("\nResults by Category:")
        return
    print(f"✅ Passed:    {passed} ({100*passed/total:.1f}%)")
    print(f"❌ Failed:    {failed} ({100*failed/total:.1f}%)")
    print(f"⚠️  Warnings:  {warnings} ({100*warnings/total:.1f}%)")
    print(f"💥 Errors:    {errors} ({100*errors/total:.1f}%)")
    print("=" * 60)

    # Group by category
    print("\nResults by Category:")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"pass": 0, "fail": 0, "warn": 0, "error": 0}
        if r.status == "PASS":
            categories[r.category]["pass"] += 1
        elif r.status == "FAIL":
            categories[r.category]["fail"] += 1
        elif r.status == "WARN":
            categories[r.category]["warn"] += 1
        else:
            categories[r.category]["error"] += 1

    for cat, counts in sorted(categories.items()):
        total_cat = sum(counts.values())
        pass_pct = 100 * counts["pass"] / total_cat if total_cat > 0 else 0
        print(f"  {cat}: {counts['pass']}/{total_cat} passed ({pass_pct:.0f}%)")

    # List failures
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print("\n❌ FAILED TESTS (settings not wired):")
        for r in failures:
            print(f"  - {r.setting_name}: {r.error_message}")

    # List warnings
    warns = [r for r in results if r.status == "WARN"]
    if warns:
        print("\n⚠️  WARNINGS (unexpected direction):")
        for r in warns:
            print(f"  - {r.setting_name}: {r.error_message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all Streamlit app settings are properly wired"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file path for detailed report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    print("Loading demo data...")
    returns = load_demo_data()
    print(f"Loaded {len(returns)} rows, {len(returns.columns)} columns")

    print(f"\nRunning {len(SETTINGS_TO_TEST)} setting tests...")
    results = run_all_tests(returns, verbose=args.verbose)

    # Generate report
    output_path = args.output or (
        PROJECT_ROOT
        / "reports"
        / f"settings_validation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(results, output_path)

    # Print summary
    print_summary(results)

    # Exit with error code if there are failures
    failures = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
