# UI Parameter Audit and Implementation Plan

**Created:** December 6, 2025  
**Status:** In Progress  
**Last Updated:** December 6, 2025 (Phase 1 Complete)

---

## Executive Summary

This document catalogs all configurable parameters in the Trend Analysis codebase that affect simulation outcomes, tracks which are currently exposed in the Streamlit UI, and provides a phased implementation plan for adding missing parameters.

---

## Current State

| Status | Count |
|--------|-------|
| ✅ Currently in UI | **26** |
| ❌ NOT in UI | **~49** |

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
| `min_tenure_n` | PolicyConfig | ❌ | Min periods before fund removal |
| `turnover_budget_max_changes` | PolicyConfig | ❌ | Max hires+fires per period |

### CATEGORY 7: Transaction Costs

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `transaction_cost_bps` | UI model_state | ✅ | Cost per trade (bps) |
| `cost_model.slippage_bps` | defaults.yml | ❌ | Additional slippage (bps) |

### CATEGORY 8: Trend Signal Parameters

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `TrendSpec.window` | TrendSpec dataclass | ❌ | Rolling signal window |
| `TrendSpec.lag` | TrendSpec dataclass | ❌ | Signal lag (delay) |
| `TrendSpec.min_periods` | TrendSpec dataclass | ❌ | Min periods for rolling calc |
| `TrendSpec.vol_adjust` | TrendSpec dataclass | ❌ | Vol-adjust signals |
| `TrendSpec.vol_target` | TrendSpec dataclass | ❌ | Signal vol target |
| `TrendSpec.zscore` | TrendSpec dataclass | ❌ | Z-score normalize signals |

### CATEGORY 9: Threshold-Hold / Entry-Exit Rules

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `z_entry_soft` | replacer.py | ❌ | Z-score for soft entry |
| `z_entry_hard` | replacer.py | ❌ | Z-score for hard entry |
| `z_exit_soft` | replacer.py | ❌ | Z-score for soft exit |
| `z_exit_hard` | replacer.py | ❌ | Z-score for hard exit |
| `sticky_add_x` | PolicyConfig | ❌ | Periods required for add |
| `sticky_drop_y` | PolicyConfig | ❌ | Periods required for drop |
| `ci_level` | PolicyConfig | ❌ | Confidence interval level |

### CATEGORY 10: Robustness / Covariance Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `robustness.shrinkage.enabled` | defaults.yml | ❌ | Enable covariance shrinkage |
| `robustness.shrinkage.method` | defaults.yml | ❌ | Shrinkage method |
| `robustness.condition_check.threshold` | defaults.yml | ❌ | Condition number threshold |
| `robustness.condition_check.safe_mode` | defaults.yml | ❌ | Fallback method |

### CATEGORY 11: Diversification

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `diversification_max_per_bucket` | PolicyConfig | ❌ | Max funds per category |
| `diversification_buckets` | PolicyConfig | ❌ | Category mapping |

### CATEGORY 12: Regime Analysis

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `regime.enabled` | defaults.yml | ❌ | Enable regime detection |
| `regime.proxy` | defaults.yml | ❌ | Regime proxy (e.g., SPX) |
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
|-----------|----------|-------|-------------|
| `seed` | defaults.yml | ❌ | Random seed |
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
| `min_tenure_n` | Number input | Advanced Settings |
| `turnover_budget_max_changes` | Number input | Advanced Settings |
| `max_active` | Number input | Fund Selection |

**Implementation Notes:**
- Add to existing Advanced Settings section
- These control churn and concentration limits

**Status:** 🔴 Not Started

---

### Phase 4: Trend Signal Parameters (MEDIUM PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Expose momentum signal configuration

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `TrendSpec.window` | Number input | Signal Settings (new section) |
| `TrendSpec.lag` | Number input | Signal Settings |
| `TrendSpec.zscore` | Checkbox | Signal Settings |

**Implementation Notes:**
- Create new collapsible "Signal Settings" section
- Only show if user wants to customize (default to hidden)

**Status:** 🔴 Not Started

---

### Phase 5: Entry/Exit Thresholds (MEDIUM PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Z-score based entry/exit rules

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `z_entry_soft` | Number input | Entry/Exit Rules (new) |
| `z_exit_soft` | Number input | Entry/Exit Rules |
| `sticky_add_x` | Number input | Entry/Exit Rules |
| `sticky_drop_y` | Number input | Entry/Exit Rules |

**Implementation Notes:**
- Advanced feature, should be collapsible/hidden by default
- Requires threshold-hold policy to be active

**Status:** 🔴 Not Started

---

### Phase 6: Regime Analysis Toggle (LOW PRIORITY)
**Estimated Effort:** 1 hour  
**Goal:** Enable/disable regime-conditional behavior

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `regime.enabled` | Checkbox | Advanced Settings |
| `regime.proxy` | Selectbox | Advanced Settings (conditional) |

**Implementation Notes:**
- Simple toggle with benchmark selection
- Lower priority as not all users need this

**Status:** 🔴 Not Started

---

### Phase 7: Robustness & Expert Settings (LOW PRIORITY)
**Estimated Effort:** 2 hours  
**Goal:** Covariance matrix handling for advanced users

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `robustness.shrinkage.enabled` | Checkbox | Expert Settings |
| `robustness.shrinkage.method` | Selectbox | Expert Settings |
| `leverage_cap` | Number input | Expert Settings |
| `seed` | Number input | Expert Settings |

**Implementation Notes:**
- Hide behind "Expert Mode" toggle
- Most users should never need to touch these

**Status:** 🔴 Not Started

---

## Progress Tracking

| Phase | Description | Status | Completed |
|-------|-------------|--------|-----------|
| 1 | Date/Time Parameters | ✅ Complete | Dec 6, 2025 |
| 2 | Risk-Free Rate & Vol Controls | ✅ Complete | Dec 6, 2025 |
| 3 | Fund Holding Rules | 🔴 Not Started | - |
| 4 | Trend Signal Parameters | 🔴 Not Started | - |
| 5 | Entry/Exit Thresholds | 🔴 Not Started | - |
| 6 | Regime Analysis | 🔴 Not Started | - |
| 7 | Robustness & Expert | 🔴 Not Started | - |

---

## Changelog

**December 6, 2025 - Phase 1 & 2 Complete:**
- Added `date_mode` radio selector (relative vs explicit)
- Added `start_date` and `end_date` date pickers for explicit mode
- Added `rf_rate_annual` input (risk-free rate %)
- Added `vol_floor` input (volatility floor %)
- Added `warmup_periods` input
- Updated `analysis_runner.py` to support explicit date mode
- Updated presets with new parameters

- **2025-12-06:** Initial audit completed. 75 parameters identified, 20 currently in UI.
