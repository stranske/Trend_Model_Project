# Settings Wiring Evidence Summary

**Generated from 15 tests using real Trend Universe data**

## Overview

| Status | Count |
|--------|-------|
| ✅ PASS | 8 |
| ❌ FAIL | 7 |
| ⚠️ WARN | 0 |
| 🚫 ERROR | 0 |
| **Total** | **15** |

## Per-Category Breakdown

| Category | Total | Effective |
|----------|-------|-----------|
| Constraints | 3 | 1 |
| Costs | 1 | 0 |
| Entry/Exit | 2 | 2 |
| Multi-Period | 2 | 2 |
| Risk | 2 | 2 |
| Selection | 3 | 1 |
| Signals | 1 | 0 |
| Weighting | 1 | 0 |

## ✅ Passing Settings

| Setting | Baseline | Test | Direction |
|---------|----------|------|-----------|
| `inclusion_approach` | 0.2334 | 0.0934 | decrease |
| `z_entry_soft` | N/A | N/A | n/a |
| `z_exit_soft` | N/A | N/A | n/a |
| `min_weight` | 0.0345 | 0.0367 | increase |
| `mp_max_funds` | N/A | N/A | n/a |
| `risk_target` | 0.1315 | 0.2630 | increase |
| `lookback_periods` | 36 | 60 | increase |
| `vol_floor` | N/A | N/A | n/a |

## ❌ Non-Effective Settings (with recommendations)

| Setting | Category | Reason | Recommendation |
|---------|----------|--------|----------------|
| `leverage_cap` | Constraints | Setting had no effect on output | Ensure leverage constraints are applied to gross exposure and portfolio construction. |
| `max_weight` | Constraints | Setting had no effect on output | Check weighting logic so max position caps flow into portfolio construction. |
| `weighting_scheme` | Weighting | Setting had no effect on output | Verify weighting_scheme switches the weighting engine used for positions. |
| `transaction_cost_bps` | Costs | Setting had no effect on output | Ensure transaction costs are applied to turnover cost and net returns. |
| `trend_window` | Signals | Setting had no effect on output | Pass trend_window through signal calculation and rolling windows. |
| `rank_pct` | Selection | Setting had no effect on output | Verify rank_pct is used when top_pct selection is active. |
| `selection_count` | Selection | Setting had no effect on output | Ensure selection_count is applied in selection logic and pipeline. |
