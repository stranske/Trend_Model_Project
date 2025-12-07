# UI Parameter Audit and Implementation Plan

**Created:** December 6, 2025  
**Status:** Phases 1-16 Complete ✅ (FINAL)  
**Last Updated:** December 7, 2025

---

## Executive Summary

This document catalogs all configurable parameters in the Trend Analysis codebase that affect simulation outcomes, tracks which are currently exposed in the Streamlit UI, and provides a phased implementation plan for adding missing parameters.

---

## Current State

| Status | Count |
|--------|-------|
| ✅ Currently in UI | **76** (70 top-level + 6 metric weights) |
| ⏭️ Excluded (duplicative/internal) | **6** |
| **Total Identified** | **~82** |

**Progress:** Started with ~20 parameters in UI, now have 76 (+280% increase). All meaningful parameters implemented.

---

## Complete Parameter Inventory

### CATEGORY 1: Time Period / Date Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `date_mode` | UI model_state | ✅ **NEW** | Relative vs explicit date mode |
| `start_date` | UI model_state | ✅ **NEW** | Simulation start date (explicit mode) |
| `end_date` | UI model_state | ✅ **NEW** | Simulation end date (explicit mode) |
| `sample_split.date` | defaults.yml | ⏭️ **SKIP** | Duplicative with date_mode/start_date/end_date |
| `lookback_months` | UI model_state | ✅ | In-sample lookback window |
| `evaluation_months` | UI model_state | ✅ | Out-of-sample evaluation window |
| `multi_period.in_sample_len` | defaults.yml | ✅ **Phase 8** | Rolling in-sample length (years) |
| `multi_period.out_sample_len` | defaults.yml | ✅ **Phase 8** | Rolling out-sample length (years) |
| `multi_period.frequency` | defaults.yml | ✅ **Phase 8** | Period frequency (M/Q/A) |

### CATEGORY 2: Fund Selection Parameters

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `selection_count` (top_k) | UI model_state | ✅ | Number of funds to select |
| `min_history_months` (min_track_months) | UI model_state | ✅ | Minimum track record required |
| `rank.inclusion_approach` | defaults.yml | ✅ **Phase 8** | top_n / top_pct / threshold |
| `rank.pct` | defaults.yml | ✅ **Phase 9** | Percentage for top_pct approach |
| `rank.threshold` | defaults.yml | ✅ **Phase 9** | Z-score threshold for threshold approach |
| `rank.score_by` | defaults.yml | ⏭️ **SKIP** | Metric for scoring (implicit via blended weights) |
| `rank.transform` | defaults.yml | ✅ **Phase 8** | Score transformation (zscore/rank/none) |
| `bottom_k` | PolicyConfig | ✅ **Phase 8** | Number of bottom funds to exclude |
| `multi_period.frequency` | defaults.yml | ✅ **Phase 8** | Multi-period rolling frequency |
| `multi_period.in_sample_len` | defaults.yml | ✅ **Phase 8** | In-sample window length |
| `multi_period.out_sample_len` | defaults.yml | ✅ **Phase 8** | Out-sample window length |
| `multi_period.min_funds` | defaults.yml | ✅ **Phase 12** | Minimum funds per period |
| `multi_period.max_funds` | defaults.yml | ✅ **Phase 12** | Maximum funds per period |

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
| `max_active_positions` | UI model_state | ✅ **Phase 3** | Max active positions |
| `constraints.long_only` | defaults.yml | ✅ **Phase 15** | Long-only constraint |
| `constraints.group_caps` | defaults.yml | ⏭️ **SKIP** | Caps per group/sector (complex) |
| `leverage_cap` | UI model_state | ✅ **Phase 7** | Max gross exposure |
| `weighting.shrink_tau` | defaults.yml | ⏭️ **SKIP** | Bayesian shrinkage parameter (niche) |
| `selector.threshold` | defaults.yml | ⏭️ **SKIP** | Duplicative with rank.threshold |

### CATEGORY 5: Risk / Volatility Settings

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `risk_target` (target_vol) | UI model_state | ✅ | Target portfolio volatility |
| `rf_rate_annual` | UI model_state | ✅ **NEW** | Risk-free rate for Sharpe calc |
| `vol_floor` | UI model_state | ✅ **NEW** | Minimum volatility floor |
| `warmup_periods` | UI model_state | ✅ **NEW** | Initial zero-weight periods |
| `vol_adjust.enabled` | defaults.yml | ✅ **Phase 10** | Enable volatility adjustment |
| `vol_adjust.window.length` | defaults.yml | ✅ **Phase 10** | Rolling vol window length |
| `vol_adjust.window.decay` | defaults.yml | ✅ **Phase 10** | EWMA vs simple |
| `vol_adjust.window.lambda` | defaults.yml | ✅ **Phase 10** | EWMA decay factor |

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
| `cost_model.slippage_bps` | defaults.yml | ✅ **Phase 8** | Additional slippage (bps) |

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
| `z_entry_hard` | replacer.py | ✅ **Phase 13** | Z-score for hard entry (immediate action) |
| `z_exit_soft` | UI model_state | ✅ **NEW** | Z-score for soft exit |
| `z_exit_hard` | replacer.py | ✅ **Phase 13** | Z-score for hard exit (immediate action) |
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
| `robustness.condition_check.threshold` | defaults.yml | ✅ **Phase 14** | Condition number threshold |
| `robustness.condition_check.safe_mode` | defaults.yml | ✅ **Phase 14** | Fallback method |

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
| `regime.method` | defaults.yml | ✅ **Phase 11** | Detection method |
| `regime.lookback` | defaults.yml | ✅ **Phase 11** | Regime lookback window |
| `regime.threshold` | defaults.yml | ✅ **Phase 11** | Regime change threshold |

### CATEGORY 13: Data / Preprocessing

| Parameter | Location | In UI | Description |
|-----------|----------|-------|-------------|
| `data.frequency` | defaults.yml | ⏭️ **SKIP** | Data frequency (auto-detected from upload) |
| `data.missing_policy` | defaults.yml | ✅ **Phase 16** | Missing data handling |
| `data.risk_free_column` | defaults.yml | ⏭️ **SKIP** | Risk-free rate column (uses rf_rate_annual) |
| `preprocessing.winsorise.enabled` | defaults.yml | ✅ **Phase 16** | Enable winsorization |
| `preprocessing.winsorise.limits` | defaults.yml | ✅ **Phase 16** | Winsorization limits |

### CATEGORY 14: Randomization

| Parameter | Location | In UI | Description |
|-----------|----------|-------|---------|
| `random_seed` | UI model_state | ✅ **NEW** | Random seed |

---

## Excluded Parameters (Internal/Duplicative)

The following parameters are excluded from the UI as they are internal system settings or duplicative:

| Parameter | Reason |
|-----------|--------|
| `n_jobs` | Internal parallelization setting - not a user analysis parameter |
| `rank.score_by` | Duplicative - implicit via blended metric weights |
| `sample_split.date` | Duplicative - covered by date_mode/start_date/end_date |
| `selector.threshold` | Duplicative - covered by rank.threshold |
| `diversification_buckets` | Requires category mapping data structure - complex |
| `diversification_max_per_bucket` | Only useful with buckets - defer |
| `weighting.shrink_tau` | Very advanced Bayesian parameter - niche |
| `constraints.group_caps` | Requires category mapping - complex |

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

### Phase 8: Multi-Period & Selection Settings (MEDIUM PRIORITY)
**Estimated Effort:** 2-3 hours  
**Goal:** Enable rolling walk-forward analysis and configurable selection approaches

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `multi_period_enabled` | Checkbox | Multi-Period Settings |
| `multi_period_frequency` | Selectbox (M/Q/A) | Multi-Period Settings |
| `in_sample_years` | Number input | Multi-Period Settings |
| `out_sample_years` | Number input | Multi-Period Settings |
| `inclusion_approach` | Selectbox (top_n/top_pct/threshold) | Selection Settings |
| `rank_transform` | Selectbox (none/zscore/rank) | Selection Settings |
| `slippage_bps` | Number input | Cost Settings |
| `bottom_k` | Number input | Exclusion Settings |

**Implementation Notes:**
- Added collapsible "Multi-Period & Selection Settings" section
- Multi-period enables walk-forward analysis with configurable windows
- Selection approach allows top N, top percentage, or z-score threshold methods
- Score transforms: raw, z-score normalized, or percentile rank
- Slippage adds market impact cost beyond transaction costs
- Bottom K excludes worst-ranked funds from selection pool
- Updated PRESET_CONFIGS with values per preset

**Status:** ✅ Complete

---

### Phase 10: Volatility Adjustment Details (MEDIUM PRIORITY)
**Estimated Effort:** 1-2 hours  
**Goal:** Fine-tune volatility scaling behavior

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `vol_adjust.enabled` | Checkbox | Vol Adjustment |
| `vol_adjust.window.length` | Number input (default 63) | Vol Adjustment |
| `vol_adjust.window.decay` | Selectbox (ewma/simple) | Vol Adjustment |
| `vol_adjust.window.lambda` | Number input (default 0.94) | Vol Adjustment |

**Implementation Notes:**
- Add to existing Risk Settings or new collapsible section
- Lambda input only shown when decay=ewma
- Update analysis_runner to pass vol_adjust config

**Status:** ✅ Complete

---

### Phase 11: Extended Regime Settings (MEDIUM PRIORITY)
**Estimated Effort:** 1 hour  
**Goal:** Full regime detection configuration

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `regime.method` | Selectbox (rolling_return/volatility) | Regime Analysis |
| `regime.lookback` | Number input (default 126) | Regime Analysis |
| `regime.threshold` | Number input (default 0.0) | Regime Analysis |

**Implementation Notes:**
- Extends existing Regime Analysis section
- Only shown when regime_enabled=True
- Update analysis_runner regime config

**Status:** ✅ Complete

---

### Phase 12: Multi-Period Bounds (LOW PRIORITY)
**Estimated Effort:** 30 min  
**Goal:** Guardrails for walk-forward analysis

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `multi_period.min_funds` | Number input (default 10) | Multi-Period Settings |
| `multi_period.max_funds` | Number input (default 25) | Multi-Period Settings |

**Implementation Notes:**
- Add to existing Multi-Period section
- Only shown when multi_period_enabled=True
- Update analysis_runner multi_period config

**Status:** ✅ Complete

---

### Phase 13: Hard Entry/Exit Thresholds (MEDIUM PRIORITY)
**Estimated Effort:** 1 hour  
**Goal:** Immediate action thresholds for extreme signals

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `z_entry_hard` | Number input | Entry/Exit Rules |
| `z_exit_hard` | Number input | Entry/Exit Rules |

**Implementation Notes:**
- Add to existing Entry/Exit Rules section
- Hard thresholds trigger immediate action (vs soft which require strikes)
- Typically set more extreme than soft (e.g., entry_hard=2.0, exit_hard=-2.0)
- Update analysis_runner threshold_hold config

**Status:** ✅ Complete

---

### Phase 14: Robustness Fallbacks (LOW PRIORITY)
**Estimated Effort:** 30 min  
**Goal:** Matrix stability fallback settings

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `robustness.condition_check.threshold` | Number input (default 1e12) | Expert Settings |
| `robustness.condition_check.safe_mode` | Selectbox (hrp/risk_parity/diagonal_mv) | Expert Settings |

**Implementation Notes:**
- Add to existing Expert Settings section
- Advanced users only - controls fallback when covariance ill-conditioned
- Update analysis_runner robustness config

**Status:** ✅ Complete

---

### Phase 15: Constraints (LOW PRIORITY)
**Estimated Effort:** 15 min  
**Goal:** Portfolio constraint toggle

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `constraints.long_only` | Checkbox (default True) | Expert Settings |

**Implementation Notes:**
- Add to Expert Settings
- Currently always True but exposing allows future flexibility
- Update analysis_runner constraints config

**Status:** ✅ Complete

---

### Phase 16: Data/Preprocessing Settings (LOW PRIORITY)
**Estimated Effort:** 1 hour  
**Goal:** Data handling configuration

| Parameter | UI Element | Section |
|-----------|------------|---------|
| `data.frequency` | Selectbox (D/W/M) | Data Settings |
| `data.missing_policy` | Selectbox (drop/ffill/zero) | Data Settings |
| `data.risk_free_column` | Selectbox from columns | Data Settings |
| `preprocessing.winsorise.limits` | Two number inputs (lower/upper) | Data Settings |

**Implementation Notes:**
- Add new collapsible "Data Settings" section on Upload page or Model page
- Winsorize limits typically [0.01, 0.99]
- Update analysis_runner preprocessing config

**Status:** ✅ Complete

---

## Progress Tracking

| Phase | Description | Status | Completed |
|-------|-------------|--------|-----------|
| 1 | Date/Time Parameters | ✅ Complete | Dec 6, 2025 |
| 2 | Risk-Free Rate & Vol Controls | ✅ Complete | Dec 6, 2025 |
| 3 | Fund Holding Rules | ✅ Complete | Dec 7, 2025 |
| 4 | Trend Signal Parameters | ✅ Complete | Dec 7, 2025 |
| 5 | Entry/Exit Thresholds | ✅ Complete | Dec 7, 2025 |
| 6 | Regime Analysis | ✅ Complete | Dec 7, 2025 |
| 7 | Robustness & Expert | ✅ Complete | Dec 7, 2025 |
| 8 | Multi-Period & Selection | ✅ Complete | Dec 7, 2025 |
| 9 | Selection Approach Details | ✅ Complete | Dec 7, 2025 |
| 10 | Volatility Adjustment Details | ✅ Complete | Dec 7, 2025 |
| 11 | Extended Regime Settings | ✅ Complete | Dec 7, 2025 |
| 12 | Multi-Period Bounds | ✅ Complete | Dec 7, 2025 |
| 13 | Hard Entry/Exit Thresholds | ✅ Complete | Dec 7, 2025 |
| 14 | Robustness Fallbacks | ✅ Complete | Dec 7, 2025 |
| 15 | Constraints (long_only) | ✅ Complete | Dec 7, 2025 |
| 16 | Data/Preprocessing Settings | ✅ Complete | Dec 7, 2025 |

---

## Changelog

**December 7, 2025 - Phases 10-16 Complete (FINAL):**
- **Phase 10: Volatility Adjustment Details**
  - Added `vol_adjust_enabled` checkbox to enable/disable vol scaling
  - Added `vol_window_length` input for rolling window size (default 63)
  - Added `vol_window_decay` dropdown (EWMA vs Simple)
  - Added `vol_ewma_lambda` input for EWMA decay factor (default 0.94)
  - New collapsible "Volatility Adjustment Details" section

- **Phase 11: Extended Regime Settings**
  - Added `regime_method` dropdown (Rolling Return vs Volatility)
  - Added `regime_lookback` input for lookback window (default 126)
  - Added `regime_threshold` input for classification threshold (default 0.0)
  - Parameters shown only when regime_enabled is True

- **Phase 12: Multi-Period Bounds**
  - Added `mp_min_funds` input for minimum funds per period (default 10)
  - Added `mp_max_funds` input for maximum funds per period (default 25)
  - Parameters shown only when multi_period_enabled is True

- **Phase 13: Hard Entry/Exit Thresholds**
  - Added `z_entry_hard` optional input for immediate entry threshold
  - Added `z_exit_hard` optional input for immediate exit threshold
  - Checkboxes to enable/disable each hard threshold
  - Hard thresholds bypass the soft strike system

- **Phase 14: Robustness Fallbacks**
  - Added `condition_threshold` input (default 1e12)
  - Added `safe_mode` dropdown (HRP, Risk Parity, Equal Weight)
  - Controls fallback behavior when covariance matrix is ill-conditioned

- **Phase 15: Constraints**
  - Added `long_only` checkbox (default True)
  - Allows users to enable/disable short selling

- **Phase 16: Data/Preprocessing**
  - Added `missing_policy` dropdown (Drop, Forward Fill, Replace with Zero)
  - Added `winsorize_enabled` checkbox
  - Added `winsorize_lower` and `winsorize_upper` inputs for percentile limits
  - Controls extreme value clipping in preprocessing

**Total Parameters Now Exposed: 76 (from ~20 originally, +280% increase)**

**December 7, 2025 - Phase 9 Complete:**
- **Phase 9: Selection Approach Details**
  - Added `rank_pct` conditional input (shown when inclusion_approach = "top_pct")
  - Added `rank_threshold` conditional input (shown when inclusion_approach = "threshold")
  - UI dynamically shows relevant input based on selected inclusion approach
  - Updated PRESET_CONFIGS with Phase 9 parameter values (varied by preset)
  - Updated analysis_runner to pass `pct` and `threshold` to portfolio.rank config

**December 7, 2025 - Phase 8 Complete:**
- **Phase 8: Multi-Period & Selection Settings**
  - Added `multi_period_enabled` checkbox for walk-forward analysis
  - Added `multi_period_frequency` dropdown (Monthly, Quarterly, Annual)
  - Added `in_sample_years` and `out_sample_years` inputs for rolling windows
  - Added `inclusion_approach` dropdown (Top N, Top Percentage, Z-Score Threshold)
  - Added `rank_transform` dropdown (None, Z-Score, Percentile Rank)
  - Added `slippage_bps` input for market impact costs
  - Added `bottom_k` input to exclude worst-ranked funds
  - New collapsible "Multi-Period & Selection Settings" section
  - Updated all presets with Phase 8 parameter values

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
