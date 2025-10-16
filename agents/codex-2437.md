<!-- bootstrap for codex on issue #2437 -->

# Scope & Constraints

> _Update 2026-11-04:_ Issue #2651 consolidated the self-test wrappers into `selftest-runner.yml`. _Update 2027-02:_ The reusable matrix is now part of `selftest-runner.yml`; retain these notes for historical reference when reviewing earlier commits.

- Limit updates to the historical self-test workflows documented in issue #2437: `selftest-83-pr-comment.yml`, `selftest-84-reusable-ci.yml`, `selftest-88-reusable-ci.yml`, `selftest-82-pr-comment.yml`, `selftest-80-pr-comment.yml`, and `selftest-81-reusable-ci.yml`. The active entry point is `selftest-runner.yml`.
- Preserve each workflow's ability to run via `workflow_dispatch` and, where already present, optional scheduled triggers (weekly cadence preferred for drift detection).
- Avoid touching unrelated workflows or altering job logic beyond the trigger configuration; no behavioural regression in reusable components.
- Ensure repository automation policies (branch protections, required checks) remain satisfied—self-tests must not start automatically on `pull_request` or generic `push` events.

# Acceptance Criteria / Definition of Done

1. All listed self-test workflow files contain only manual (`workflow_dispatch`) and, if desired, low-frequency `schedule` triggers.
2. `pull_request`, `pull_request_target`, `push`, or other automatic triggers are removed from these workflows.
3. YAML syntax remains valid and workflows still reference their reusable jobs without modification.
4. Documentation (commit/PR summary) states the manual-only scope so maintainers understand the new invocation pattern.

# Initial Task Checklist

- [x] Inventory each targeted workflow to confirm current triggers.
- [x] Remove automatic triggers, retaining or adding `workflow_dispatch` as needed.
- [x] Standardise (or reintroduce, if missing) a weekly `schedule` trigger where it provides value for drift detection.
- [x] Validate workflows locally using `act` or GitHub's workflow syntax checker (if available) to ensure no YAML mistakes.
- [x] Update changelog or operations notes if the team tracks workflow trigger adjustments.
