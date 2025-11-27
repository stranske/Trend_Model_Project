# Keepalive Status — PR #3803

> **Status:** In progress — selector cache scoped helper and lifecycle tests implemented.

## Progress updates
- Round 1: Captured scope, tasks, and acceptance criteria from the PR description for the selector cache hardening work.
- Round 2: Added a scoped cache helper with eviction limits, reset hooks, and tests covering reuse, eviction, and scope isolation.

## Scope
- [x] Prevent stale or unbounded selector window metric caches by adding scoped control and eviction support.

## Tasks
- [x] Encapsulate the window metric cache behind a helper with explicit reset/eviction and optional size bounds.
- [x] Ensure cache state can be scoped per run or cleared between analyses to avoid cross-request pollution.
- [x] Add lightweight tests covering cache hits, evictions, and reset behavior.

## Acceptance criteria
- [x] Ranking functions can reset or scope cache state, preventing stale metrics from influencing subsequent runs.
- [x] Cache size does not grow unbounded during repeated executions in tests.
- [x] New tests cover cache lifecycle behaviors.
