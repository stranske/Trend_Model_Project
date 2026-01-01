# Keepalive Status — PR #4063

> **Status:** In progress — hard threshold wiring and barrier enforcement.

## Progress updates
- Round 1: Reviewed PR #4063 changes for hard threshold wiring in multi-period selection and tests.
- Round 2: Protected low-weight removals with `z_exit_hard`, added wiring coverage, and ran `MPLCONFIGDIR=/home/runner/work/Trend_Model_Project/Trend_Model_Project/.cache/mpl TREND_ROLLING_CACHE=/home/runner/work/Trend_Model_Project/Trend_Model_Project/.cache/rolling pytest -p no:rerunfailures tests/test_multi_period_exits_cooldown.py -m "not slow"`.

## Scope
The hard threshold settings (`z_entry_hard`, `z_exit_hard`) are defined in the UI but have no implementation. These should provide absolute barriers:
- `z_entry_hard`: Funds below this z-score are NEVER selected, regardless of other criteria
- `z_exit_hard`: Funds above this z-score are NEVER removed, regardless of other criteria

## Tasks
- [x] Verify settings are passed through Config to selection logic
- [x] Implement hard entry barrier in fund selection
- [x] Implement hard exit protection in fund removal
- [x] Add unit tests for hard threshold behavior
- [x] Add wiring tests to verify settings have effect

## Acceptance criteria
- [x] Funds below `z_entry_hard` are never selected
- [x] Funds above `z_exit_hard` are never removed
- [x] Hard thresholds override soft threshold decisions
- [x] Settings wiring tests pass
- [ ] Existing soft threshold behavior unchanged
