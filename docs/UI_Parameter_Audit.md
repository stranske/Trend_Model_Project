# UI Parameter Audit and Implementation Plan

**Created:** December 6, 2025  
**Status:** Phases 1-7 Complete  
**Last Updated:** December 7, 2025

---

## Executive Summary

This document catalogs all configurable parameters in the Trend Analysis codebase that affect simulation outcomes, tracks which are currently exposed in the Streamlit UI, and provides a phased implementation plan for adding missing parameters.

---

## Current State

| Status | Count |
|--------|-------|
| ✅ Currently in UI | **40+** |
| ❌ NOT in UI (advanced) | **~35** |

---

## Complete Parameter Inventory

### CATEGORY 1: Time Period / Date Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `date_mode` | UI model_state | ✅ **NEW** | Relative vs explicit date mode |
| `start_date` | UI model_state | ✅ **NEW** | Simulation start date (explicit mode) |
| `end_date` | UI model_state | ✅ **NEW** | Simulation end date (explicit mode) |
| `sample_split.date` | defaults.yml | ❌ | In-sample / out-of-sample split date |
| `lookback_months` | UI model_state | ✅ | In-sample lookback window |
| `evaluation_months` | UI model_state | ✅ | Out-of-sample evaluation window |
| `multi_period.in_sample_len` | defaults.yml | ❌ | Rolling in-sample length (years) |
| `multi_period.out_sample_len` | defaults.yml | ❌ | Rolling out-sample length (years) |
| `multi_period.frequency` | defaults.yml | ❌ | Period frequency (M/Q/A) |

### CATEGORY 2: Fund Selection Parameters

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `selection_count` (top_k) | UI model_state | ✅ | Number of funds to select |
| `min_history_months` (min_track_months) | UI model_state | ✅ | Minimum track record required |
| `rank.inclusion_approach` | defaults.yml | ❌ | top_n / top_pct / threshold |
| `rank.pct` | defaults.yml | ❌ | Percentage for top_pct approach |
| `rank.threshold` | defaults.yml | ❌ | Z-score threshold for threshold approach |
| `rank.score_by` | defaults.yml | ❌ | Metric for scoring (implicit via weights) |
| `rank.transform` | defaults.yml | ❌ | Score transformation (zscore/rank/none) |
| `bottom_k` | PolicyConfig | ❌ | Number of bottom funds to exclude |
| `max_active` | PolicyConfig | ❌ | Maximum active positions |
| `multi_period.min_funds` | defaults.yml | ❌ | Minimum funds per period |
| `multi_period.max_funds` | defaults.yml | ❌ | Maximum funds per period |

### CATEGORY 3: Metric Weights for Ranking

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `sharpe` weight | UI model_state | ✅ | Sharpe ratio weight |
| `return_ann` weight | UI model_state | ✅ | Annual return weight |
| `sortino` weight | UI model_state | ✅ | Sortino ratio weight |
| `info_ratio` weight | UI model_state | ✅ | Information ratio weight |
| `drawdown` weight | UI model_state | ✅ | Max drawdown weight |
| `vol` weight | UI model_state | ✅ | Volatility weight |
| `info_ratio_benchmark` | UI model_state | ✅ | Benchmark for info ratio |

### CATEGORY 4: Portfolio Weighting / Construction

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `weighting_scheme` | UI model_state | ✅ | Weighting method |
| `max_weight` | UI model_state | ✅ | Max weight per fund |
| `constraints.long_only` | defaults.yml | ❌ | Long-only constraint |
| `constraints.group_caps` | defaults.yml | ❌ | Caps per group/sector |
| `leverage_cap` | defaults.yml | ❌ | Max gross exposure |
| `weighting.shrink_tau` | defaults.yml | ❌ | Bayesian shrinkage parameter |
| `selector.threshold` | defaults.yml | ❌ | Z-score selector threshold |

### CATEGORY 5: Risk / Volatility Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `risk_target` (target_vol) | UI model_state | ✅ | Target portfolio volatility |
| `rf_rate_annual` | UI model_state | ✅ **NEW** | Risk-free rate for Sharpe calc |
| `vol_floor` | UI model_state | ✅ **NEW** | Minimum volatility floor |
| `warmup_periods` | UI model_state | ✅ **NEW** | Initial zero-weight periods |
| `vol_adjust.enabled` | defaults.yml | ❌ | Enable volatility adjustment |
| `vol_adjust.window.length` | defaults.yml | ❌ | Rolling vol window length |
| `vol_adjust.window.decay` | defaults.yml | ❌ | EWMA vs simple |
| `vol_adjust.window.lambda` | defaults.yml | ❌ | EWMA decay factor |

### CATEGORY 6: Rebalancing / Turnover

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `rebalance_freq` | UI model_state | ✅ | Rebalance frequency (M/Q/A) |
| `max_turnover` | UI model_state | ✅ | Maximum turnover per period |
| `cooldown_months` | UI model_state | ✅ | Cooldown after fund removal |
| `min_tenure_periods` | UI model_state | ✅ **NEW** | Min periods before fund removal |
| `max_changes_per_period` | UI model_state | ✅ **NEW** | Max hires+fires per period |

### CATEGORY 7: Transaction Costs

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `transaction_cost_bps` | UI model_state | ✅ | Cost per trade (bps) |
| `cost_model.slippage_bps` | defaults.yml | ❌ | Additional slippage (bps) |

### CATEGORY 8: Trend Signal Parameters

| Parameter | Location | In UI | Description |
|-----------|----------|-------|---------|
| `trend_window` | UI model_state | ✅ **NEW** | Rolling signal window |
| `trend_lag` | UI model_state | ✅ **NEW** | Signal lag (delay) |
| `trend_min_periods` | UI model_state | ✅ **NEW** | Min periods for rolling calc |
| `trend_vol_adjust` | UI model_state | ✅ **NEW** | Vol-adjust signals |
| `trend_vol_target` | UI model_state | ✅ **NEW** | Signal vol target |
| `trend_zscore` | UI model_state | ✅ **NEW** | Z-score normalize signals |

### CATEGORY 9: Threshold-Hold / Entry-Exit Rules

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `z_entry_soft` | UI model_state | ✅ **NEW** | Z-score for soft entry |
| `z_entry_hard` | replacer.py | ❌ | Z-score for hard entry (not commonly used) |
| `z_exit_soft` | UI model_state | ✅ **NEW** | Z-score for soft exit |
| `z_exit_hard` | replacer.py | ❌ | Z-score for hard exit (not commonly used) |
| `soft_strikes` | UI model_state | ✅ **NEW** | Consecutive exit periods |
| `entry_soft_strikes` | UI model_state | ✅ **NEW** | Consecutive entry periods |
| `sticky_add_periods` | UI model_state | ✅ **NEW** | Periods required for add |
| `sticky_drop_periods` | UI model_state | ✅ **NEW** | Periods required for drop |
| `ci_level` | UI model_state | ✅ **NEW** | Confidence interval level |

### CATEGORY 10: Robustness / Covariance Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|---------|
| `shrinkage_enabled` | UI model_state | ✅ **NEW** | Enable covariance shrinkage |
| `shrinkage_method` | UI model_state | ✅ **NEW** | Shrinkage method |
| `leverage_cap` | UI model_state | ✅ **NEW** | Maximum leverage cap |
| `robustness.condition_check.threshold` | defaults.yml | ❌ | Condition number threshold |
| `robustness.condition_check.safe_mode` | defaults.yml | ❌ | Fallback method |

### CATEGORY 11: Diversification

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `diversification_max_per_bucket` | PolicyConfig | ❌ | Max funds per category |
| `diversification_buckets` | PolicyConfig | ❌ | Category mapping |

### CATEGORY 12: Regime Analysis

| Parameter | Location | In UI | Description |
|-----------|----------|-------|---------|
| `regime_enabled` | UI model_state | ✅ **NEW** | Enable regime detection |
| `regime_proxy` | UI model_state | ✅ **NEW** | Regime proxy (e.g., SPX) |
| `regime.method` | defaults.yml | ❌ | Detection method |
| `regime.lookback` | defaults.yml | ❌ | Regime lookback window |
| `regime.threshold` | defaults.yml | ❌ | Regime change threshold |

### CATEGORY 13: Data / Preprocessing

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `data.frequency` | defaults.yml | ❌ | Data frequency (D/W/M) |
| `data.missing_policy` | defaults.yml | ❌ | Missing data handling |
| `data.risk_free_column` | defaults.yml | ❌ | Risk-free rate column |
| `preprocessing.winsorise.limits` | defaults.yml | ❌ | Winsorization limits |

### CATEGORY 14: Execution / Randomization

| Parameter | Location | In UI | Description |
|-----------|----------|-------|---------|
| `random_seed` | UI model_state | ✅ **NEW** | Random seed |
| `n_jobs` | defaults.yml | ❌ | Parallel jobs |

---

## Implementation Plan

### Phase 1: Critical Date/Time Parameters (HIGH PRIORITY)
**Estimated Effort:** 2-3 hours  
**Goal:** Allow users to specify simulation date ranges

| Parameter | UI Element | Section |
|-----------|------------|---------|
| In-sample start date | Date picker | Time Period Settings |
| In-sample end date | Date picker | Time Period Settings |
| Out-of-sample start date | Date picker (auto-calculated) | Time Period Settings |
| Out-of-sample end date | Date picker | Time Period Settings |

**Implementation Notes:**
- Add new "Time Period Settings" section at top of Model page
- Auto-calculate IS end = OOS start - 1 day based on lookback
- Validate dates against uploaded data range
- Show data coverage indicator

**Status:** ✅ Complete

---

### Phase 2: Risk-Free Rate and Volatility Controls (HIGH PRIORITY)
**Estimated Effort:** 1-2 hours  
**Goal:** Control Sharpe calculation inputs and vol adjustment

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `rf_rate_annual` | Number input (%) | Risk Settings |
| `vol_adjust.enabled` | Checkbox | Risk Settings |
| `vol_adjust.floor_vol` | Number input (%) | Risk Settings |
| `warmup_periods` | Number input | Advanced Settings |
| `slippage_bps` | Number input | Advanced Settings |

**Implementation Notes:**
- Add RF rate input near target volatility
- Add vol adjustment toggle with conditional floor_vol input
- Update analysis_runner to pass these to Config

**Status:** ✅ Complete

---

### Phase 3: Fund Holding Rules (MEDIUM PRIORITY)
**Estimated Effort:** 1-2 hours  
**Goal:** Fine-tune fund addition/removal behavior

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `min_tenure_periods` | Number input | Fund Holding Rules |
| `max_changes_per_period` | Number input | Fund Holding Rules |
| `max_active_positions` | Number input | Fund Holding Rules |

**Implementation Notes:**
- Added new "Fund Holding Rules" section
- Controls churn and concentration limits
- Added to PRESET_CONFIGS with values per preset

**Status:** ✅ Complete

---

### Phase 4: Trend Signal Parameters (MEDIUM PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Expose momentum signal configuration

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `trend_window` | Number input | Signal Settings |
| `trend_lag` | Number input | Signal Settings |
| `trend_min_periods` | Number input (optional) | Signal Settings |
| `trend_zscore` | Checkbox | Signal Settings |
| `trend_vol_adjust` | Checkbox | Signal Settings |
| `trend_vol_target` | Number input (%) | Signal Settings |

**Implementation Notes:**
- Created collapsible "Signal Settings" section
- Builds TrendSpec from UI parameters in analysis_runner
- Added to PRESET_CONFIGS with values per preset

**Status:** ✅ Complete

---

### Phase 5: Entry/Exit Thresholds (MEDIUM PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Z-score based entry/exit rules

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `z_entry_soft` | Number input | Entry/Exit Rules |
| `z_exit_soft` | Number input | Entry/Exit Rules |
| `soft_strikes` | Number input | Entry/Exit Rules |
| `entry_soft_strikes` | Number input | Entry/Exit Rules |
| `sticky_add_periods` | Number input | Entry/Exit Rules |
| `sticky_drop_periods` | Number input | Entry/Exit Rules |
| `ci_level` | Slider (0-0.99) | Entry/Exit Rules |

**Implementation Notes:**
- Added collapsible "Entry/Exit Rules" section
- Z-score thresholds control manager hiring/firing based on relative performance
- Soft strikes require consecutive periods below threshold before removal
- Sticky periods require consistent ranking before hiring/firing
- CI level adds confidence interval gate for entry decisions
- Integrated with threshold_hold policy in portfolio config
- Values passed to PolicyConfig for simulator

**Status:** ✅ Complete

---

### Phase 6: Regime Analysis Toggle (LOW PRIORITY)
**Estimated Effort:** 1 hour  
**Goal:** Enable/disable regime-conditional behavior

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `regime_enabled` | Checkbox | Regime Analysis |
| `regime_proxy` | Selectbox | Regime Analysis |

**Implementation Notes:**
- Added collapsible "Regime Analysis" section
- Proxy dropdown shows available benchmarks from data
- Passes to Config.regime dict in analysis_runner

**Status:** ✅ Complete

---

### Phase 7: Robustness & Expert Settings (LOW PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Covariance matrix handling for advanced users

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `shrinkage_enabled` | Checkbox | Expert Settings |
| `shrinkage_method` | Selectbox | Expert Settings |
| `leverage_cap` | Number input | Expert Settings |
| `random_seed` | Number input | Expert Settings |

**Implementation Notes:**
- Added collapsible "Expert Settings" section
- Shrinkage methods: Ledoit-Wolf, OAS, None
- Leverage cap limits gross exposure
- Random seed for reproducibility

**Status:** ✅ Complete

---

## Progress Tracking

| Phase | Description | Status | Completed |
|-------|-------------|--------|-----------||
| 1 | Date/Time Parameters | ✅ Complete | Dec 6, 2025 |
| 2 | Risk-Free Rate & Vol Controls | ✅ Complete | Dec 6, 2025 |
| 3 | Fund Holding Rules | ✅ Complete | Dec 7, 2025 |
| 4 | Trend Signal Parameters | ✅ Complete | Dec 7, 2025 |
| 5 | Entry/Exit Thresholds | ✅ Complete | Dec 7, 2025 |
| 6 | Regime Analysis | ✅ Complete | Dec 7, 2025 |
| 7 | Robustness & Expert | ✅ Complete | Dec 7, 2025 |

---

## Changelog

**December 7, 2025 - Phase 5 Complete:**
- **Phase 5: Entry/Exit Thresholds**
  - Added `z_entry_soft` and `z_exit_soft` number inputs for z-score thresholds
  - Added `soft_strikes` and `entry_soft_strikes` for consecutive period requirements
  - Added `sticky_add_periods` and `sticky_drop_periods` for ranking persistence
  - Added `ci_level` slider for confidence interval gate
  - New collapsible "Entry/Exit Rules" section
  - Integrated with threshold_hold policy in portfolio config
  - Updated analysis_runner to pass thresholds to Config
  - Controls manager hiring and firing decision logic

**December 7, 2025 - Phases 3, 4, 6, 7 Complete:**
- **Phase 3: Fund Holding Rules**
  - Added `min_tenure_periods` input
  - Added `max_changes_per_period` input  
  - Added `max_active_positions` input
  - New "Fund Holding Rules" section in UI
  
- **Phase 4: Trend Signal Parameters**
  - Added `trend_window`, `trend_lag`, `trend_min_periods` inputs
  - Added `trend_zscore`, `trend_vol_adjust` checkboxes
  - Added `trend_vol_target` input
  - New collapsible "Signal Settings" section
  - Updated analysis_runner to build TrendSpec from UI params
  
- **Phase 6: Regime Analysis**
  - Added `regime_enabled` checkbox
  - Added `regime_proxy` dropdown (uses benchmark columns from data)
  - New collapsible "Regime Analysis" section
  
- **Phase 7: Robustness & Expert Settings**
  - Added `shrinkage_enabled` checkbox and `shrinkage_method` dropdown
  - Added `leverage_cap` input
  - Added `random_seed` input
  - New collapsible "Expert Settings" section
  - Updated analysis_runner with robustness config

**December 6, 2025 - Phase 1 & 2 Complete:**
- Added `date_mode` radio selector (relative vs explicit)
- Added `start_date` and `end_date` date pickers for explicit mode
- Added `rf_rate_annual` input (risk-free rate %)
- Added `vol_floor` input (volatility floor %)
- Added `warmup_periods` input
- Updated `analysis_runner.py` to support explicit date mode
- Updated presets with new parameters

**December 6, 2025:**
- Initial audit completed. 75 parameters identified, 20 currently in UI.
