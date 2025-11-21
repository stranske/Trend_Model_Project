<!-- bootstrap for codex on issue #3683 -->

## Scope
- [ ] Create `portfolio/weight_policy.py` with a function that:
  - [ ] Drops assets with invalid signals OR carries last valid weight OR reallocates to cash, based on a config flag.
  - [ ] Normalizes only after filtering and checks a “min assets” threshold before allocation.
- [ ] Apply this function wherever weights are built.
- [ ] Unit tests for each policy, including a warm-up period.

## Tasks
- [ ] Implement the policy function and wire it into portfolio construction.
- [ ] Tests for drop/carry/reallocate branches and “min assets” behavior.

## Acceptance criteria
- [ ] No NaNs in resulting weights/returns.
- [ ] Weight sums remain stable given each policy’s rules.
