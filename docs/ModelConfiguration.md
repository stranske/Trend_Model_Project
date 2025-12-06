# Model Configuration Reference

This document provides detailed explanations of all configuration parameters available on the **Model** page in the Streamlit application. Each setting controls how the trend-following portfolio analysis is performed.

---

## Quick Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Preset | Baseline | — | Pre-configured signal settings |
| Window | 63 | 1+ months | Lookback period for trend calculation |
| Lag | 1 | 1–window | Delay before acting on signals |
| Minimum Periods | 63 | 1–window | Required data points for valid signal |
| Volatility Adjust | Off | On/Off | Scale returns by volatility |
| Volatility Target | 0.10 | 0.0+ | Target annualized volatility |
| Row Z-Score | Off | On/Off | Normalize signals cross-sectionally |
| Lookback Months | 36 | 12+ | In-sample training period |
| Evaluation Window | 12 | 3+ months | Out-of-sample test period |
| Selection Count | 10 | 1+ | Number of funds to include |
| Weighting Scheme | equal | equal, vol_target | How to allocate across funds |
| Metric Weights | 1.0 each | 0.0+ | Relative importance of each metric |
| Target Volatility | 0.10 | 0.0+ | Portfolio volatility target |

---

## Trend Signal Settings

### Preset

**What it does:** Selects a pre-configured set of signal parameters optimized for different market conditions.

**Options:**
- **Baseline** – Default settings suitable for most analyses. Uses a 63-month window with no volatility adjustment.
- **Conservative** – Longer lookback windows and higher minimum periods reduce noise but may lag market turns.
- **Aggressive** – Shorter windows respond faster to trends but increase signal noise.
- **Custom** – Manually configure all parameters below.

**When to change:** Use Conservative in volatile or uncertain markets. Use Aggressive when you believe trends are short-lived and want faster responsiveness.

---

### Window (months)

**What it does:** The number of months of historical returns used to calculate the trend signal. This is the "lookback period" for the moving average or momentum calculation.

**Default:** 63 months (~5 years)

**Trade-offs:**
- **Longer windows (60–120):** Smoother signals, fewer false positives, but slower to react to regime changes
- **Shorter windows (12–36):** More responsive to recent trends, but noisier and prone to whipsaws

**Example:** A 36-month window bases the trend signal on the past 3 years of returns.

---

### Lag

**What it does:** Number of months to delay before acting on a trend signal. Helps avoid reacting to very recent noise.

**Default:** 1 month

**Range:** 1 to Window length

**When to increase:** If you observe excessive turnover from reacting to short-term reversals, increasing lag to 2–3 months can smooth portfolio changes.

---

### Minimum Periods

**What it does:** The minimum number of valid data points required before a trend signal is calculated. Funds with less history produce no signal until this threshold is met.

**Default:** Same as Window

**Trade-offs:**
- **Set equal to Window:** Strictest; only funds with complete history get signals
- **Set lower (e.g., Window/2):** Allow signals from newer funds with partial history

**Example:** With Window=36 and Min Periods=18, a fund with only 20 months of data will still get a trend signal.

---

### Volatility Adjust

**What it does:** When enabled, returns are scaled by their historical volatility before computing trend signals. This makes signals comparable across assets with different volatility profiles.

**Default:** Off

**When to enable:**
- Comparing funds with very different volatility levels (e.g., equity vs bond funds)
- When you want trend strength to be volatility-normalized

**Requires:** Set a Volatility Target > 0 when enabled.

---

### Volatility Target

**What it does:** The target annualized volatility used when scaling returns for volatility-adjusted trend signals.

**Default:** 0.10 (10% annualized)

**Only active when:** Volatility Adjust is enabled

**How it works:** If a fund has 20% annualized volatility and the target is 10%, its returns are scaled by 0.5 before computing the trend signal.

---

### Row Z-Score

**What it does:** Normalizes trend signals cross-sectionally at each time point. Each fund's signal becomes a z-score relative to the mean and standard deviation of all funds at that date.

**Default:** Off

**When to enable:**
- When you want to rank funds purely by relative momentum
- Useful in sector-rotation or relative-value strategies

**Trade-off:** Removes absolute signal magnitude; a fund can have a positive z-score even if its absolute momentum is negative.

---

## Portfolio Settings

### Lookback Months (In-Sample Period)

**What it does:** Defines the length of the "training" or "in-sample" period used to calculate fund metrics (Sharpe ratio, returns, drawdowns) for ranking and selection.

**Default:** 36 months (3 years)

**Trade-offs:**
- **Longer (48–60 months):** More statistical significance, but may include stale data from different regimes
- **Shorter (12–24 months):** More relevant to current conditions, but higher estimation error

**Relationship:** The in-sample period ends right before the evaluation window begins.

---

### Evaluation Window (Out-of-Sample Period)

**What it does:** Defines the "test" or "out-of-sample" period over which portfolio performance is measured after fund selection.

**Default:** 12 months

**How it works:** Funds are selected based on their in-sample metrics, then held for the evaluation period to measure realized performance.

**Minimum:** 3 months (shorter periods have high variance)

---

### Selection Count

**What it does:** The number of top-ranked funds to include in the portfolio.

**Default:** 10 funds

**Trade-offs:**
- **Fewer funds (3–5):** Concentrated portfolio, higher tracking to selected funds, more sensitive to individual fund performance
- **More funds (15–20):** Diversified portfolio, lower individual fund impact, closer to index-like behavior

**Constraint:** Cannot exceed the number of available funds in your dataset.

---

### Weighting Scheme

**What it does:** Determines how capital is allocated across the selected funds.

**Options:**

| Scheme | Description |
|--------|-------------|
| **equal** | Each fund receives equal weight (1/N allocation). Simple, robust, and easy to understand. |
| **vol_target** | Weights are inversely proportional to volatility, targeting equal risk contribution. Higher-volatility funds get lower weights. |

**When to use vol_target:** When you want to balance risk across funds rather than capital. Particularly useful when selected funds have very different volatility profiles.

---

## Metric Weights

### Sharpe Ratio Weight

**What it does:** Controls the importance of risk-adjusted returns (Sharpe ratio) in the fund ranking score.

**Default:** 1.0

**Sharpe Ratio formula:** (Annualized Return − Risk-Free Rate) / Annualized Volatility

**When to increase:** Prioritize consistent risk-adjusted performance over raw returns.

---

### Annual Return Weight

**What it does:** Controls the importance of raw annualized returns in the fund ranking score.

**Default:** 1.0

**When to increase:** Prioritize funds with highest absolute returns, regardless of volatility.

---

### Max Drawdown Weight

**What it does:** Controls the importance of maximum drawdown (largest peak-to-trough decline) in the fund ranking score.

**Default:** 1.0

**Note:** Lower (less negative) drawdowns are better. The ranking penalizes funds with large drawdowns.

**When to increase:** Prioritize capital preservation and reduce exposure to funds with historical large losses.

---

### How Metric Weights Combine

The final ranking score is a weighted average:

```
Score = (w_sharpe × Sharpe_zscore) + (w_return × Return_zscore) + (w_dd × Drawdown_zscore)
```

Where:
- Each metric is z-scored across all funds
- Weights are normalized to sum to 1.0
- Higher scores = better funds

**Example:** Weights of Sharpe=2.0, Return=1.0, Drawdown=1.0 means Sharpe ratio counts twice as much as each of the other metrics.

---

## Risk Settings

### Target Volatility (Portfolio)

**What it does:** The target annualized volatility for the overall portfolio. The system scales portfolio weights to achieve approximately this volatility level.

**Default:** 0.10 (10% annualized)

**How it works:** If the unscaled portfolio has 15% volatility and the target is 10%, weights are reduced by 1/3 (with the remainder in cash).

**Trade-offs:**
- **Lower targets (5–8%):** Conservative, more cash, smaller drawdowns
- **Higher targets (12–20%):** Aggressive, fully invested or leveraged, larger potential gains and losses

---

## Tips for Common Scenarios

### Conservative Long-Term Investor
- Preset: Conservative or Baseline
- Window: 60+ months
- Selection Count: 15–20
- Metric Weights: Emphasize Sharpe and Max Drawdown
- Target Volatility: 6–8%

### Aggressive Trend Follower
- Preset: Aggressive
- Window: 24–36 months
- Selection Count: 5–8
- Metric Weights: Emphasize Annual Return
- Target Volatility: 15–20%

### Balanced Portfolio
- Preset: Baseline
- Window: 36–48 months
- Selection Count: 10–12
- Metric Weights: Equal weights (1.0 each)
- Target Volatility: 10%

---

## See Also

- [User Guide](UserGuide.md) – Overview of all application features
- [Preset Strategies](PresetStrategies.md) – Details on Conservative, Balanced, and Aggressive presets
- [Weighting Options](UserGuide.md#6-weighting-options) – Full reference for weighting engines
- [Risk Controls](UserGuide.md#7-risk-controls) – Advanced volatility and constraint settings
