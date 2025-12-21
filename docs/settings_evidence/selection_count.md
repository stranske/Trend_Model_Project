# Setting: `selection_count`

**Test Date:** 2025-12-21T12:49:58.020727
**Status:** FAIL

## Configuration

- **Baseline Value:** `10`
- **Test Value:** `5`
- **Expected Metric:** `num_funds_selected`
- **Expected Direction:** `decrease`

## Results

- **Baseline Metric:** 68
- **Test Metric:** 68
- **Metric Changed:** False
- **Direction Correct:** False

**Note:** Setting had no effect on output

## Economic Interpretation

The number of funds in the portfolio changed from baseline to test. With baseline=68 and test=68, the portfolio became more diversified. Economic intuition: More funds typically reduce idiosyncratic risk but may dilute alpha.