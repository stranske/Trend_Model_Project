# Setting: `max_weight`

**Test Date:** 2025-12-21T12:55:56.099559
**Status:** FAIL

## Configuration

- **Baseline Value:** `0.2`
- **Test Value:** `0.1`
- **Expected Metric:** `max_position_weight`
- **Expected Direction:** `decrease`

## Results

- **Baseline Metric:** 1.3741
- **Test Metric:** 1.3741
- **Metric Changed:** False
- **Direction Correct:** False

**Note:** Setting had no effect on output

## Economic Interpretation

Maximum position size constraint. Baseline=1.3741, Test=1.3741. Looser constraints on concentration. Economic intuition: Lower max weights force diversification, potentially reducing volatility.