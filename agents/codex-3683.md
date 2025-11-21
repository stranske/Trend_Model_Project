<!-- bootstrap for codex on issue #3683 -->

## Scope
- [x] Create `portfolio/weight_policy.py` with a function that:
  - [x] Drops assets with invalid signals OR carries last valid weight OR reallocates to cash, based on a config flag.
  - [x] Normalizes only after filtering and checks a “min assets” threshold before allocation.
- [x] Apply this function wherever weights are built.
- [x] Unit tests for each policy, including a warm-up period.

## Tasks
- [x] Implement the policy function and wire it into portfolio construction.
- [x] Tests for drop/carry/reallocate branches and “min assets” behavior.

## Acceptance criteria
- [x] No NaNs in resulting weights/returns.
- [x] Weight sums remain stable given each policy’s rules.
