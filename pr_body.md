<!-- pr-preamble:start -->
> **Source:** Issue #4135

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The `max_weight` constraint in `compute_constrained_weights()` is applied AFTER volatility scaling, but the constraint logic assumes weights sum to 1.0. When vol targeting scales up low-volatility assets, individual weights can exceed 1.0 (e.g., 2.0 per asset), causing the constraint to fail with `ConstraintViolation: max_weight too small for number of assets`. The pipeline then falls back to base weights, completely ignoring the max_weight constraint.

#### Tasks
- [x] Identify correct ordering of constraint application vs vol scaling in `src/trend_analysis/risk.py`
- [x] Apply max_weight constraint to normalized weights (sum=1.0) before vol scaling
- [x] Ensure constraint still enforces intended behavior after scaling
- [x] Add test case that verifies max_weight works with vol_adjust enabled
- [x] Update settings wiring test for max_weight to use vol_adjust (currently disabled as workaround)

#### Acceptance criteria
- [x] `max_weight=0.35` with `vol_adjust_enabled=True` produces weights capped at 35%
- [x] No `ConstraintViolation` exceptions when both settings are enabled
- [x] Settings effectiveness test for max_weight passes without disabling vol_adjust
- [x] Unit tests verify constraint + scaling interaction

<!-- auto-status-summary:end -->
