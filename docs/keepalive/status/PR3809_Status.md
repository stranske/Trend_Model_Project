# Keepalive Status — PR #3809

> **Status:** Not started — recorded initial scope, tasks, and acceptance criteria for the single-period guardrails and correlation surfacing work.

## Progress updates
- Round 1: Registered the PR's scope, tasks, and acceptance criteria from the bootstrap instructions; no implementation steps taken yet.

## Scope
- [ ] Guard `single_period_run` window handling so empty or all-NaN slices return explicit diagnostics instead of silent NaNs.
- [ ] Surface correlation metric errors with actionable messages rather than suppressing them.
- [ ] Expand coverage for empty-window and all-NaN scenarios to validate the guarded behaviors.

## Tasks
- [ ] Add guards that detect empty or all-NaN window slices and emit a clear error or warning instead of propagating NaN metrics.
- [ ] Tighten error handling around optional metrics (e.g., correlations) to log or raise informative exceptions instead of silent pass-through.
- [ ] Expand tests to cover empty-window and all-NaN scenarios, asserting clear failure modes.

## Acceptance criteria
- [ ] Running `single_period_run` on empty or all-NaN windows produces explicit diagnostics and avoids silent NaNs in outputs.
- [ ] Correlation metric failures are surfaced with actionable messages.
- [ ] New tests validate the guarded behaviors.
