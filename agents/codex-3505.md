<!-- bootstrap for codex on issue #3505 -->

## Scope
- Enforce the keepalive orchestrator run cap directly inside the PR-meta workflow before any dispatch happens.
- Mirror the same run-cap guard inside the orchestrator so a late-arriving workflow instance cannot post an extra instruction.
- Keep the cap default at 2 and allow overrides via the existing `agents:max-runs:X` label (X ∈ 1..5).

## Non-Goals
- Changing orchestrator concurrency groups or worker behavior beyond the explicit run-cap guard.
- Renaming labels, altering Gate contents, or modifying branch-sync logic.

## Tasks
- [ ] Implement the PR-meta run-cap check in `.github/workflows/agents-pr-meta.yml` just ahead of the dispatch step.
  - [ ] Gather orchestrator runs for this PR with statuses `queued` and `in_progress` and count them as `active`.
  - [ ] Resolve the cap from the `agents:max-runs:X` label (default 2 when absent) and expose `ok = active < cap`.
  - [ ] When `ok` is false, skip dispatch and emit the single-line summary `CAP: ok=false reason=run-cap-reached pr=#<n> cap=<cap> active=<active>`.
- [ ] Apply the same guard inside `.github/workflows/agents-70-orchestrator.yml` before posting the next keepalive instruction.
  - [ ] Reuse the PR-meta counting logic (queued + in_progress) so orchestrator instances honor the cap consistently.
  - [ ] Ensure the orchestrator summary records the skip with the same format as PR-meta when the cap is reached.
- [ ] Add or update automated coverage (unit or integration) so the new guard paths are exercised and regressions surface.
- [ ] Provide evidence (logs or test output) demonstrating the guard allows up to the cap, blocks the extra dispatch, and leaves normal sequences unaffected.

## Acceptance Criteria
- [ ] On a test PR with `agents:max-runs:2`, two near-simultaneous dispatch attempts succeed and a third is skipped with the documented `CAP:` summary line.
- [ ] Orchestrator runs continue to execute in parallel up to the configured cap; no unintended serialization occurs.
- [ ] The Checks list for the PR no longer shows bursts above the cap, confirming the guard blocks the extra dispatch.
- [ ] Automated coverage exercises both the allowed and skipped paths for the run-cap guard.
