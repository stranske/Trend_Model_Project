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

## ✅ Passing Settings

| Setting | Baseline | Test | Direction |
|---------|----------|------|-----------|
| `inclusion_approach` | 0.2334 | 0.0934 | decrease |
| `z_entry_soft` | 8066ee3f | 22b897e1 | - |
| `z_exit_soft` | 8066ee3f | 757e5c44 | - |
| `min_weight` | 0.0345 | 0.0367 | increase |
| `mp_max_funds` | 8066ee3f | 77047489 | - |
| `risk_target` | 0.1315 | 0.2630 | increase |
| `lookback_periods` | 36 | 60 | increase |
| `vol_floor` | 8066ee3f | 67ccfd8e | - |

## ❌ Failing Settings (Investigation Required)

| Setting | Reason |
|---------|--------|
| `leverage_cap` | Setting had no effect on output |
| `max_weight` | Setting had no effect on output |
| `weighting_scheme` | Setting had no effect on output |
| `transaction_cost_bps` | Setting had no effect on output |
| `trend_window` | Setting had no effect on output |
| `rank_pct` | Setting had no effect on output |
| `selection_count` | Setting had no effect on output |

