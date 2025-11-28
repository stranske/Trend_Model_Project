# Keepalive Status — PR #3827

> **Status:** In progress — recording the scope, tasks, and acceptance criteria so keepalive nudges continue until completion.

## Progress updates
- Round 1: Captured the PR scope, tasks, and acceptance criteria for keepalive tracking.

## Scope
- [ ] Signal computation uses only window-scoped data, eliminating look-ahead bias in both in-sample and out-of-sample runs.
- [ ] Tests cover both windows and assert failures when future data is present.
- [ ] Rebalance outputs align strictly to window boundaries without forward-looking inputs.

## Tasks
- [ ] Scope every signal calculation to the active analysis window before any reindexing so no future observations influence the period being scored.
- [ ] Add guardrails/tests that fail fast when signal helpers receive dates outside the requested slice.
- [ ] Audit each rebalance/reindex step to ensure signals and weights stay aligned to data available at the rebalance date.

## Acceptance criteria
- [ ] Signal computation uses only window-scoped data, eliminating look-ahead bias in both in-sample and out-of-sample runs.
- [ ] Tests cover both windows and assert failures when future data is present.
- [ ] Rebalance outputs align strictly to window boundaries without forward-looking inputs.
