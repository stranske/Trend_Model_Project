<!-- Coordination notes for Issue #2193 -->

# Issue #2193 – Self-Test Workflow Consolidation

## Scope
- Remove legacy self-test callers (`selftest-82-pr-comment.yml`, `maint-43-*.yml`, `maint-44-*.yml`, `maint-48-*.yml`, `selftest-80-pr-comment.yml`).
- Retain a single human-facing caller (`selftest-80-reusable-ci.yml`) that triggers only on manual dispatch.
- Keep the `selftest-8X-*` series manual-only; do not reintroduce cron triggers.
- Update docs/tests to reflect the new roster so regressions are caught automatically.

## Acceptance Criteria
- ✅ Only `selftest-80-reusable-ci.yml` appears in the Actions tab for self-tests.
- ✅ The manual workflow invokes `./.github/workflows/reusable-10-ci-python.yml` via the scenario matrix job.
- ✅ No other workflow names containing “selftest” remain.
- ✅ CI guardrails assert the above invariants.

## Task Checklist
- [ ] Delete the redundant self-test workflow callers.
- [ ] Ensure the manual caller exposes only `workflow_dispatch` (no cron).
- [ ] Confirm delegation to the reusable CI composite via the scenario matrix job.
- [ ] Update documentation references (`docs/ci/WORKFLOWS.md`, audit sheets) if required.
- [ ] Add/extend automated tests so the roster cannot regress.
- [ ] Run workflow guardrail tests (`pytest tests/test_workflow_*.py`).

## Notes
- Post-CI summary and failure tracker workflows now rely solely on manual dispatch; keep the workflow title `Selftest 80 Reusable CI Matrix` aligned with filename-driven tests.
- Do not reintroduce cron triggers or extra inputs—the matrix scenarios live inside the reusable job.
