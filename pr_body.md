<!-- pr-preamble:start -->
> **Source:** Issue #4137

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The `max_turnover` setting is intended to limit portfolio turnover between rebalancing periods, but the settings effectiveness test shows it has no observable effect on actual turnover. Both baseline (max_turnover=1.0) and test (max_turnover=0.3) produce identical turnover values (1.325).

#### Tasks
- [ ] Trace `max_turnover` config value through pipeline to verify it reaches `_enforce_turnover_cap()`
- [ ] Check if turnover constraint is bypassed when natural turnover is already low
- [x] Verify `turnover` field in period_results captures constrained turnover, not desired turnover
- [ ] Fix constraint enforcement if not working correctly
- [x] Update test to use scenario that generates high natural turnover

#### Acceptance criteria
- [x] `max_turnover=0.3` produces measurably lower turnover than `max_turnover=1.0`
- [ ] Period-level turnover values respect the configured cap
- [x] Settings effectiveness test for max_turnover passes
- [x] Unit test verifies turnover constraint is applied

<!-- auto-status-summary:end -->
