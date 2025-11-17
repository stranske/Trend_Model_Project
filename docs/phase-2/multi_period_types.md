## Multi-Period Result Type Schema

This document describes the structured typing added for Phase‑2 multi‑period
back‑testing outputs. The goal is to provide a stable, explicit contract for
downstream exporters, UIs, and tests while preserving backward compatibility
with the existing dictionary-based result objects.

### `MultiPeriodPeriodResult`

Each element returned from `trend_analysis.multi_period.engine.run(cfg, df)` is
an instance of `MultiPeriodPeriodResult` (a `TypedDict`) with the following
fields. Optional keys MAY be absent; consumers must code defensively.

| Key | Type | Required | Description |
| --- | ---- | -------- | ----------- |
| `period` | `tuple[str, str, str, str]` | Yes | `(in_start, in_end, out_start, out_end)` month strings (YYYY-MM) delimiting the analysis windows. |
| `out_ew_stats` | `Mapping[str, float]` | Yes | Equal‑weighted portfolio statistics for the out‑of‑sample window. |
| `out_user_stats` | `Mapping[str, float]` | Yes | User‑weighted (custom) portfolio statistics for the out‑of‑sample window. |
| `manager_changes` | `list[dict[str, object]]` | Threshold‑hold only | Sequence of change events (seed, added, dropped, replacement, low_weight_strikes, z_entry, z_exit). |
| `turnover` | `float` | Threshold‑hold only | Period turnover (L1 weight change). |
| `transaction_cost` | `float` | Threshold‑hold only | Turnover * ((transaction_cost_bps + slippage_bps) / 10,000). |
| `cov_diag` | `list[float]` | Optional | Diagonal of covariance matrix (diagnostic / perf flags). |
| `cache_stats` | `Mapping[str, int | float]` | Optional | Performance cache metrics (when covariance cache enabled). |

All remaining keys propagate from the underlying Phase‑1 `_run_analysis` result
dictionary unchanged (e.g. `in_sample_stats`, `out_sample_stats`, raw frames, etc.)
to avoid breaking existing exporter expectations.

### Backward Compatibility

The introduction of the `TypedDict` did **not** alter runtime structure; it
purely annotates shape for static analysis. Tests were added to ensure:

1. A result object is produced for **every** generated period (alignment test).
2. Threshold‑hold branch continues to populate `manager_changes` / `turnover` / `transaction_cost`.

### Alignment Invariant

`len(results) == len(generate_periods(cfg.model_dump()))` must hold. The
regression fixed in Sept 2025 (indentation moving `results.append` outside the
loop) is now guarded by `tests/test_threshold_hold_alignment.py`.

### Event Log (`manager_changes`)

Each event dict minimally includes: `action`, `manager`, `firm`, `reason`, and
`detail`. Reasons presently include: `seed`, `replacement`, `low_weight_strikes`,
`one_per_firm`, `rebalance`, `z_entry`, `z_exit`.

### Future Extensions

Planned optional fields (not yet implemented) may include:

* `drawdown_path`: Per-period drawdown time‑series for UI hover tooltips.
* `risk_budget`: Allocation vs. risk-budget share diagnostics.
* `selector_log`: Structured selection diagnostics (rank values, thresholds).

Any additions will remain optional keys to prevent breaking existing
deserialisers. Exporters should always access with `.get()`.

### Usage Notes

Downstream code should treat values as immutable analysis artifacts:

```python
from trend_analysis.multi_period import run as run_multi

results = run_multi(cfg, returns_df)
for period_result in results:
    in_start, in_end, out_start, out_end = period_result["period"]
    user_stats = period_result["out_user_stats"]
    sharpe = user_stats.get("sharpe")
    changes = period_result.get("manager_changes", [])
    # ... consumption logic ...
```

---

Maintainer Note: Keep this document in sync when adding new optional keys or
altering the Phase‑2 multi‑period export contract.
