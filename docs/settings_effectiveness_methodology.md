# Settings Effectiveness Evaluation Methodology

This document describes the methodology used to evaluate whether UI settings in the Streamlit app produce meaningful changes in simulation outputs.

## Overview

The settings effectiveness evaluation framework tests each configurable setting by:
1. Running a baseline simulation with default settings
2. Running a test simulation with one setting changed
3. Comparing outputs to detect meaningful differences
4. Categorizing results and generating recommendations

## Key Concepts

### Effectiveness Rate

The **effectiveness rate** measures what percentage of settings produce detectable output changes:

```
Effectiveness Rate = (PASS + WARN) / Total Settings
```

- **Target**: ≥80% effectiveness rate
- **PASS**: Setting change produced expected effect
- **WARN**: Setting change produced effect, but in unexpected direction (still counts as "effective")
- **FAIL**: No detectable effect
- **ERROR**: Test could not complete

### Categories

Settings are grouped into categories for analysis:

| Category | Description |
|----------|-------------|
| `portfolio` | Portfolio construction settings (weights, selection) |
| `risk` | Risk management settings (vol target, floor) |
| `timing` | Time-related settings (windows, periods) |
| `signals` | Signal generation settings (trend, z-scores) |
| `cost` | Transaction cost settings |
| `multi_period` | Multi-period analysis settings |
| `weighting` | Weight calculation settings |

### Mode-Specific Settings

Some settings only have effect when specific modes are enabled:

- `rank_pct` only affects output when `inclusion_approach="top_pct"`
- `selection_count` only affects output when `inclusion_approach="top_n"`
- `buy_hold_initial` only affects output when `inclusion_approach="buy_and_hold"`

These are tested with the appropriate mode context enabled.

## Test Methodology

### 1. Baseline State

Each test starts with a consistent baseline state derived from:
- The "Baseline" preset configuration
- Standard defaults for all settings
- Mode-specific overrides where needed

### 2. Setting Variation

For each setting, we determine an appropriate test value:

| Setting Type | Variation Strategy |
|--------------|-------------------|
| Numeric | Increase/decrease by meaningful amount |
| Boolean | Toggle from baseline |
| Enum/Choice | Select next option in list |
| Percentage | Adjust by 5-10 percentage points |

### 3. Output Comparison

We compare multiple metrics between baseline and test runs:

**Portfolio Metrics:**
- Weight L1 distance (sum of absolute weight differences)
- Maximum weight change
- Number of positions changed

**Return Metrics:**
- Mean return difference
- Total return difference
- Return volatility difference
- Sharpe ratio difference

**Risk Metrics:**
- CAGR difference
- Volatility difference
- Maximum drawdown difference
- Tracking error vs baseline

**Statistical Significance:**
- Sign-flip test p-value (< 0.05 for significance)

### 4. Classification Rules

A setting is classified based on these rules:

```python
if reporting_only_setting:
    status = "NO_EFFECT"  # Expected for display-only settings
elif metric_changed and (significant or p_value_invalid):
    status = "EFFECTIVE" if no_mode_context else "MODE_SPECIFIC"
else:
    status = "NO_EFFECT"  # Setting not wired
```

## Report Outputs

### CSV Report

Contains detailed per-setting results:
- Setting name and category
- Baseline and test values
- All comparison metrics
- Status and recommendation

### Effectiveness JSON

Summary statistics including:
- Overall effectiveness rate
- Per-category breakdown
- List of non-effective settings with reasons
- Recommendations for fixing

### Console Summary

Real-time output showing:
- Pass/fail/warn/error counts
- Effectiveness rate vs target
- Per-category effectiveness
- Non-effective settings with recommendations

## Recommendations System

For each non-effective setting, the framework generates actionable recommendations:

| Pattern | Recommendation |
|---------|---------------|
| Mode-related setting | Verify correct prerequisite settings |
| Weight-related setting | Check weighting logic in metrics.py |
| Time-window setting | Ensure passed through pipeline |
| Generic | Check if wired from UI to analysis |

## Running the Evaluation

```bash
# Full evaluation with all outputs
python scripts/test_settings_wiring.py -o reports/settings_report.csv

# With coverage comparison to model page
python scripts/test_settings_wiring.py --coverage-report reports/coverage.json

# Verbose mode for debugging
python scripts/test_settings_wiring.py -v
```

## CI Integration

The workflow can enforce effectiveness thresholds:

```yaml
- name: Check effectiveness threshold
  run: |
    rate=$(jq '.effectiveness_rate' reports/*.effectiveness.json)
    if (( $(echo "$rate < 0.80" | bc -l) )); then
      echo "Effectiveness rate $rate below 80% threshold"
      exit 1
    fi
```

## Maintenance

### Adding New Settings

1. Add setting to `SETTINGS_TO_TEST` list with:
   - Name matching UI key
   - Category
   - Expected metric and direction
   - Baseline and test values

2. If mode-specific, add to `MODE_CONTEXT` mapping

3. Run evaluation to verify setting is detected

### Updating Thresholds

Effectiveness threshold can be adjusted in:
- `compute_effectiveness_summary()` function (`target_rate` parameter)
- CI workflow threshold check

## Related Files

- `scripts/test_settings_wiring.py` - Main evaluation script
- `scripts/evaluate_settings_effectiveness.py` - Supporting utilities
- `docs/settings_evidence/` - Per-setting evidence files
- `reports/` - Generated reports
