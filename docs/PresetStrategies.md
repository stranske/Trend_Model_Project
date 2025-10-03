# Preset Strategies

The project ships with three configuration presets that provide sensible defaults for common risk profiles. Each preset lives in [`config/presets/`](../config/presets) and can be selected from the Streamlit app's **Configure** page or loaded on the command line.

| Preset | Risk profile | Key traits |
| --- | --- | --- |
| **Conservative** | Low risk tolerance | 60‑month lookback, quarterly rebalancing, target volatility 8%, focuses on stability metrics |
| **Balanced** | Moderate risk | 36‑month lookback, monthly rebalancing, target volatility 10%, even mix of return and risk metrics |
| **Aggressive** | High risk tolerance | 24‑month lookback, monthly rebalancing, target volatility 15%, emphasises return and agility |

The trend-signal presets follow the same spectrum. Conservative applies a 126-period
window with heavier smoothing (min 90 periods, z-scored, 8% volatility target),
Balanced uses an 84-period window with moderate smoothing and a 10% target, while
Aggressive drops to 42 periods with minimal smoothing and a 15% volatility cap.

## When to use each preset

### Conservative
Choose this when capital preservation is a priority. The longer lookback window and lower risk target favour stable managers and infrequent turnover.

### Balanced
A good starting point for most users. Provides a middle ground between stability and responsiveness with monthly rebalancing and moderate volatility targeting.

### Aggressive
Use when seeking higher returns and willing to accept larger drawdowns. Shorter lookbacks and a higher risk target make the portfolio more responsive but also more volatile.

To run the pipeline with a preset on demo data:

```bash
python -m trend_analysis.run_analysis -c config/presets/balanced.yml
```

Replace `balanced` with `conservative` or `aggressive` as needed.
