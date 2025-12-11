# Trend Signal Settings

> **Note**: This functionality is currently **hidden from the UI** because it requires **daily returns data** to be meaningful. Most users work with monthly returns, where these settings would be inappropriate.

## Overview

The Trend Signal Settings control how time-series momentum (TSMOM) signals are computed for each fund. These signals can be used for tactical allocation, timing, or weight adjustments within the portfolio.

## When to Use

**Use these settings when:**
- You have **daily returns data**
- You want to implement trend-following or momentum strategies
- You need signals for timing entry/exit or adjusting fund weights

**Do NOT use these settings when:**
- You have **monthly returns data** (a 63-period window would span 5+ years!)
- You're only doing fund selection based on historical metrics

## Parameters

### Signal Window (periods)
- **Description**: Rolling window size for computing trend signals
- **Default**: 63 (approximately 3 months of trading days)
- **Range**: 5-252 periods
- **Config key**: `trend_window`

With daily data:
- 21 periods ≈ 1 month
- 63 periods ≈ 3 months
- 126 periods ≈ 6 months
- 252 periods ≈ 1 year

### Signal Lag
- **Description**: Number of periods to lag the signal for causality
- **Default**: 1
- **Range**: 1-10 periods
- **Config key**: `trend_lag`

A lag of at least 1 is required to avoid look-ahead bias.

### Min Periods
- **Description**: Minimum observations required before computing a valid signal
- **Default**: None (uses full window)
- **Range**: 0-252
- **Config key**: `trend_min_periods`

Set to 0 or leave blank to require the full window before producing signals.

### Cross-sectional Z-score
- **Description**: Standardize signals across all funds at each time step
- **Default**: False
- **Config key**: `trend_zscore`

When enabled, signals are converted to z-scores relative to the cross-section of all funds, making them comparable across different volatility regimes.

### Volatility Adjust Signals
- **Description**: Scale signals by volatility to normalize across assets
- **Default**: False
- **Config key**: `trend_vol_adjust`

When enabled, high-volatility funds have their signals scaled down to be comparable with low-volatility funds.

### Signal Vol Target
- **Description**: Target volatility for vol-adjusted signals
- **Default**: 0.10 (10%)
- **Range**: 0.01-0.50
- **Config key**: `trend_vol_target`

Only applicable when volatility adjustment is enabled.

## Configuration Example

```yaml
signals:
  trend:
    window: 63
    lag: 1
    min_periods: null
    zscore: false
    vol_adjust: false
    vol_target: 0.10
```

## Re-enabling in the UI

To re-enable this section in the Streamlit UI when daily returns support is added:

1. Edit `streamlit_app/pages/2_Model.py`
2. Find the comment `# Section 7: Trend Signal Settings - REMOVED FROM UI`
3. Restore the expander and input widgets (see git history for the original code)
4. Consider adding a data frequency check to show/hide based on detected frequency

## Technical Details

The trend signal computation is implemented in `src/trend_analysis/signals.py` using the `TrendSpec` dataclass and `compute_trend_signals()` function. The signals use a time-series momentum approach where:

1. Rolling returns are computed over the signal window
2. Signals are optionally volatility-adjusted
3. Signals are optionally z-scored cross-sectionally
4. A lag is applied to ensure causality
