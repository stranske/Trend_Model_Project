## Datetime Frequency Policy

As of September 2025 Pandas has deprecated certain ambiguous legacy frequency
aliases for *timestamp* ranges (e.g. `pd.date_range(freq="M")` now emits a
FutureWarning recommending explicit `ME`). However, `pd.period_range` **still**
requires the legacy aliases (`"M"`, `"Q"`). Attempting to use `"ME"` or `"QE"`
with `period_range` raises:

```
ValueError: Invalid frequency: ME, failed to parse with error message: ValueError("for Period, please use 'M' instead of 'ME'")
```

### Policy Summary

| Purpose / Index Type | Monthly Alias | Quarterly Alias | Rationale |
|----------------------|---------------|-----------------|-----------|
| `pd.date_range` (Timestamp) | `ME` | `QE` | Avoid deprecation warnings; explicit *end* anchored alias |
| `pd.period_range` (Period)  | `M`  | `Q`  | Required by Pandas (ME/QE invalid for Period indices) |

### Enforced Helpers

Use constants in `trend_analysis.timefreq` instead of scattering string
literals:

```python
from trend_analysis.timefreq import (
    MONTHLY_DATE_FREQ, MONTHLY_PERIOD_FREQ,
    QUARTERLY_DATE_FREQ, QUARTERLY_PERIOD_FREQ,
    monthly_date_range, monthly_period_range,
)

idx_ts = monthly_date_range("2020-01-31", periods=6)          # date_range with ME
idx_pr = monthly_period_range("2020-01", periods=6)           # period_range with M
```

### Static Test Guard

`tests/test_no_invalid_period_freq_aliases.py` scans the repository and fails
CI if it finds `period_range(..., freq="ME")` or `period_range(..., freq="QE")`.

### Do / Don’t Quick Reference

```python
# ✅ Correct (timestamp index)
pd.date_range("2024-01-31", periods=3, freq="ME")

# ✅ Correct (period index)
pd.period_range("2024-01", periods=3, freq="M")

# ❌ Incorrect (raises ValueError)
pd.period_range("2024-01", periods=3, freq="ME")
```

### Future Change Note

If/when Pandas adds `ME`/`QE` support to `period_range`, update the constants
and relax the scanning test in one commit so behaviour remains consistent.
