<!-- bootstrap for codex on issue #2437 -->

# Scope & Constraints

- Limit updates to the self-test workflows documented in issue #2437: `maint-43-selftest-pr-comment.yml`, `maint-44-selftest-reusable-ci.yml`, `maint-48-selftest-reusable-ci.yml`, `pr-20-selftest-pr-comment.yml`, `selftest-pr-comment.yml`, and `selftest-reusable-ci.yml`.
- Preserve each workflow's ability to run via `workflow_dispatch` and, where already present, optional scheduled triggers (weekly cadence preferred for drift detection).
- Avoid touching unrelated workflows or altering job logic beyond the trigger configuration; no behavioural regression in reusable components.
- Ensure repository automation policies (branch protections, required checks) remain satisfied—self-tests must not start automatically on `pull_request` or generic `push` events.

# Acceptance Criteria / Definition of Done

1. All listed self-test workflow files contain only manual (`workflow_dispatch`) and, if desired, low-frequency `schedule` triggers.
2. `pull_request`, `pull_request_target`, `push`, or other automatic triggers are removed from these workflows.
3. YAML syntax remains valid and workflows still reference their reusable jobs without modification.
4. Documentation (commit/PR summary) states the manual-only scope so maintainers understand the new invocation pattern.

# Initial Task Checklist

- [ ] Inventory each targeted workflow to confirm current triggers.
- [ ] Remove automatic triggers, retaining or adding `workflow_dispatch` as needed.
- [ ] Standardise (or reintroduce, if missing) a weekly `schedule` trigger where it provides value for drift detection.
- [ ] Validate workflows locally using `act` or GitHub's workflow syntax checker (if available) to ensure no YAML mistakes.
- [ ] Update changelog or operations notes if the team tracks workflow trigger adjustments.
