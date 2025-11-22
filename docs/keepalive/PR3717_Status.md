# Keepalive Status — PR #3717

> **Status:** Lambda-based turnover penalty implemented and tested; all checklist items satisfied.

## Scope
- [x] In portfolio construction, add optional parameter `lambda_tc` that shrinks proposed weight changes toward previous weights (simple L1-style penalty or convex combination).
- [x] Switchable without touching signal definitions.
- [x] Unit tests on a toy series confirming reduced turnover as `lambda_tc` increases.

## Tasks
- [x] Implement penalty logic gated by config.
- [x] Add tests demonstrating monotone decrease in turnover vs. `lambda_tc`.

## Acceptance criteria
- [x] With the same signals, higher `lambda_tc` yields lower turnover in tests.
