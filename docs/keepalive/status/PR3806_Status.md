# Keepalive Status — PR #3806

> **Status:** In progress — captured initial scope, tasks, and acceptance criteria for the risk-free fallback updates.

## Progress updates
- Round 1: Recorded the PR's scope, tasks, and acceptance criteria for the risk-free fallback and alignment work.

## Scope
- [ ] Stabilize risk-free handling in `rank_select_funds`, enabling a documented fallback path that aligns with the requested analysis window.

## Tasks
- [ ] Document and enable a safe default behavior when no risk-free column is supplied, with an explicit configuration flag.
- [ ] Align fallback detection with the requested analysis window so volatility comparisons use the same slice as downstream metrics.
- [ ] Add tests that cover missing columns, fallback paths, and window alignment to ensure deterministic selection.

## Acceptance criteria
- [ ] `rank_select_funds` no longer hard-fails by default when the risk-free column is omitted but follows a documented fallback when enabled.
- [ ] Fallback selection uses window-aligned data and produces deterministic results in tests.
- [ ] Unit tests cover both explicit column selection and fallback scenarios.
