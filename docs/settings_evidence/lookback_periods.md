# Setting: `lookback_periods`

**Test Date:** 2025-12-21T12:12:13.608373
**Status:** PASS

## Configuration

- **Baseline Value:** `3`
- **Test Value:** `5`
- **Expected Metric:** `in_sample_months`
- **Expected Direction:** `increase`

## Results

- **Baseline Metric:** 36
- **Test Metric:** 60
- **Metric Changed:** True
- **Actual Direction:** increase
- **Direction Correct:** True

## Economic Interpretation

In-sample evaluation window changed. Baseline=36, Test=60. Longer lookback for ranking funds. Economic intuition: Longer lookbacks capture more trend persistence but may miss regime changes.