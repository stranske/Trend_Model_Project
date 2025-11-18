# Keepalive Status — PR #3672

## Scope
- [ ] Scope section missing from source issue (cannot check because upstream issue omitted scope block; retain open status per keepalive contract).

## Tasks
- [x] Test that missing/extra columns raise validation errors. Covered by the schema guard cases in `tests/test_invariants.py::test_validation_requires_date_and_value_columns`.
- [x] Test that non-members are excluded at rebalance. Exercised via `tests/test_invariants.py::test_gate_universe_excludes_non_members_at_rebalance`.
- [x] Test that rolling helper never uses future data. Guarded by `tests/test_invariants.py::test_rolling_shifted_never_uses_future_data`.
- [x] Test that costs reduce returns by the analytical amount on a toy turnover sequence. Verified by `tests/test_invariants.py::test_transaction_costs_reduce_returns_by_expected_amount`.
- [x] Test that rebalance dates occur only on the calendar utility’s dates. Ensured by `tests/test_invariants.py::test_backtest_calendar_matches_calendar_helper`.

## Acceptance Criteria
- [x] Five named tests pass; failures point to the exact invariant that broke. Each invariant-specific test above fails with descriptive assertion messages pinpointing the violated rule, so the suite now meets the acceptance bar.
