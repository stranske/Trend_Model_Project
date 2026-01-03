<!-- pr-preamble:start -->
> **Source:** Issue #4144

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The `apply_constraints()` function in `src/trend_analysis/engine/optimizer.py` contains a duplicated cash-weight processing block (lines 200-235 and 237-278). The code includes a comment explicitly acknowledging this:

```python
# NOTE: The block below duplicates the earlier cash handling logic for legacy
# payloads that mutated the constraint object between validation passes.  The
# modern ``ConstraintSet`` implementation keeps values stable, so the duplicate
# code path is effectively unreachable during normal execution.
```

While the defensive duplicate may have historical justification, it's a maintenance hazard—any fix applied to one block must be manually applied to the other, and the logic is complex enough (feasibility checks, scaling, max_weight enforcement) that divergence is likely.

#### Tasks
- [ ] Create `_apply_cash_weight()` helper function that encapsulates lines 137-175 logic
- [ ] Define helper signature: `_apply_cash_weight(w: pd.Series, cash_weight: float, max_weight: float | None) -> pd.Series`
- [ ] Replace first cash-weight block (lines 137-175) with call to helper
- [ ] Replace second cash-weight block (lines 200-235) with call to helper
- [ ] Remove the third duplicated block (lines 237-278) since it's unreachable with modern ConstraintSet
- [ ] Add unit test that validates both passes produce identical results
- [ ] Update docstring to explain the two-pass pattern

#### Acceptance criteria
- [ ] Only one implementation of cash-weight logic exists (in the helper)
- [ ] Both passes call the same helper function
- [ ] All existing optimizer tests pass unchanged
- [ ] New test verifies helper produces correct output for edge cases (0 < cash_weight < 1, max_weight conflicts)
- [ ] Coverage on the helper is 100%

<!-- auto-status-summary:end -->
