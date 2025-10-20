# Issue 2878 — Strengthen keepalive workflow coverage with executable harness

## Task List
- [x] Extract the keepalive JavaScript into `scripts/keepalive-runner.js` and update the workflow to load the helper without altering runtime behaviour.
- [x] Build the Node-based harness under `tests/fixtures/keepalive/` that executes the helper via `vm` with mocked `core`, `github`, and workflow context.
- [x] Add JSON fixtures covering opt-out, idle-threshold, dry-run, and dedupe timelines for the harness.
- [x] Implement pytest coverage in `tests/test_keepalive_workflow.py` that validates skip logic, idle threshold enforcement, dry-run previews, and deduplication while capturing summary output.
- [x] Gate the new harness tests inside `scripts/dev_check.sh` so Tier-1 validation runs them when Node.js is available.

## Acceptance Criteria
- [x] Tests fail if enable_keepalive overrides, idle/repeat windows, or dry-run preview behaviour regresses.
- [x] Summary assertions verify target labels, agent logins, and triggered/preview counts for each scenario.
- [x] Harness remains hermetic, mocking GitHub interactions and avoiding network/PAT usage; CI reports coverage over the shared helper.
- [x] Workflow logic continues to reference the shared helper with identical dry-run vs live semantics (no production behaviour change).

## Validation
- `pytest tests/test_keepalive_workflow.py`
