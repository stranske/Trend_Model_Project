# Keepalive Status — PR #4104

> **Status:** Complete — bottom_k excludes worst-ranked funds from selection.

## Progress updates
- Round 1: Reviewed prior keepalive changes, added a rank-mode integration test asserting bottom_k excludes the five worst funds, and verified the targeted pytest run.

## Scope
The bottom_k setting should exclude the worst-performing K funds from selection but has no observable effect.

## Tasks
- [x] Trace setting from UI through Config to selection
- [x] Implement exclusion of bottom K funds from selection pool
- [x] Add wiring test to verify setting affects selection

## Acceptance criteria
- [x] bottom_k=5 excludes 5 worst-scoring funds
- [x] Excluded funds never appear in portfolio
- [x] Settings wiring test passes
