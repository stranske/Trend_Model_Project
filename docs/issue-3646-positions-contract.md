# Issue 3646 — Positions Contract Tracker

_Mismatched assumptions about the structure of portfolio positions caused incorrect weights and fills. This tracker records how the codebase now enforces a single contract and proves it with tests so the keepalive workflow can see exactly which scope, tasks, and acceptance criteria are satisfied._

## Scope

- [x] Document and enforce the canonical DataFrame contract for target positions before portfolio logic consumes them. Evidence: `normalize_positions` documents the Date-index / symbol-column / [-1, 1] weight rules and raises descriptive errors whenever inputs violate that contract, while `apply_rebalance_schedule` calls the helper before gating trades.【F:src/trend_analysis/schedules.py†L51-L170】

## Task List

- [x] Define the positions contract: index is Date; columns are symbols; values are target weights in [-1, 1] or NaN for not held. Documented verbatim in the `normalize_positions` docstring and enforced by index/column/shape validation with contract-referencing error text.【F:src/trend_analysis/schedules.py†L56-L119】
- [x] Provide `normalize_positions(df)` to enforce bounds, fill NaN with 0, and mask non-eligible assets. Helper clips to [-1, 1], fills NaN with 0.0, and reindexes to any `eligible` universe while rejecting empty universes.【F:src/trend_analysis/schedules.py†L102-L120】
- [x] Add tests for overweight, underweight, and NaN handling. `tests/test_rebalance_schedule.py` clamps 1.4/−1.6 weights, fills NaN to 0, and proves eligible masking plus dtype conversion for under/overweight cases.【F:tests/test_rebalance_schedule.py†L56-L141】

## Acceptance Criteria

- [x] Portfolio layer accepts only normalized positions; rejects invalid shapes with helpful errors. `apply_rebalance_schedule` normalizes every input, preserving the first observation, sorting when required, and raising `ValueError` whenever the rebalance calendar never overlaps the provided index. Validation errors reference the normalization contract so callers understand the required shape.【F:src/trend_analysis/schedules.py†L123-L170】
- [x] Tests show reproducible final weights for a simple two-asset toy case. `test_normalize_positions_two_asset_reproducible_weights` feeds AAA/BBB targets and asserts the final normalized weights are deterministically clipped to [-1, 1] and NaN is turned into 0 / 0.5, demonstrating reproducible outcomes.【F:tests/test_rebalance_schedule.py†L144-L157】

## Evidence & Validation Log

- 2025-11-18 – `pytest tests/test_rebalance_schedule.py`
- Add future rows here if regressions are reproduced or new verification data is collected.
