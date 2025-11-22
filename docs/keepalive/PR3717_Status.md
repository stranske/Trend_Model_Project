# Keepalive Status — PR #3717

## Scope
- [ ] In portfolio construction, add optional parameter `lambda_tc` that shrinks proposed weight changes toward previous weights (simple L1-style penalty or convex combination).
- [ ] Switchable without touching signal definitions.
- [ ] Unit tests on a toy series confirming reduced turnover as `lambda_tc` increases.

## Tasks
- [ ] Implement penalty logic gated by config.
- [ ] Add tests demonstrating monotone decrease in turnover vs. `lambda_tc`.

## Acceptance criteria
- [ ] With the same signals, higher `lambda_tc` yields lower turnover in tests.
