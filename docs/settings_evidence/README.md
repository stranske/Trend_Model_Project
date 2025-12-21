# Settings Wiring Evidence

This directory contains automatically generated evidence that validates whether
each UI setting in the Streamlit app properly affects the analysis output.

## How Evidence is Generated

1. **Data Source**: Real Trend Universe Data (`data/Trend Universe Data.csv`)
   - 404 monthly periods from 1991-07 to 2025-02
   - 75 fund/index columns
   - Uses existing `load_and_validate_file()` with `missing_policy="zero"`

2. **Test Methodology**: For each setting:
   - Run baseline analysis with default settings
   - Run test analysis with one setting changed
   - Compare a relevant metric to verify the change propagated

3. **Evidence Files**:
   - `{setting_name}.md` - Human-readable evidence report
   - `{setting_name}.json` - Machine-readable test data
   - `SUMMARY.md` - Aggregated summary of all tests

## Running Tests

```bash
# List all testable settings
python scripts/generate_settings_evidence.py --list

# Test a specific setting
python scripts/generate_settings_evidence.py -s risk_target -v

# Run all tests (takes ~15 minutes)
python scripts/generate_settings_evidence.py
```

## Interpreting Results

### ✅ PASS
The setting change produced a measurable effect in the expected direction.
Example: Increasing `risk_target` from 10% to 20% → portfolio volatility increased.

### ❌ FAIL  
The setting had no measurable effect on output. This indicates:
- The setting isn't wired through to the analysis engine, OR
- The metric being measured doesn't capture the setting's effect, OR
- The baseline configuration masks the setting's effect

### ⚠️ WARN
The setting produced a change but in an unexpected direction.
Requires investigation to determine if this is correct behavior.

## Current Status

See [SUMMARY.md](SUMMARY.md) for the latest test results.

## Known Issues

Settings that fail validation may be:
1. **Not implemented**: The UI exposes the setting but it's not connected
2. **Context-dependent**: Only applies with certain other settings (e.g., `selection_count` only matters when `inclusion_approach="top_n"`)
3. **Metric mismatch**: We're measuring the wrong output metric

## Related Files

- `scripts/generate_settings_evidence.py` - Evidence generation script
- `scripts/test_settings_wiring.py` - Core testing infrastructure
- `streamlit_app/pages/8_Validation.py` - Interactive validation UI
