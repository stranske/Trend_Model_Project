<!-- Coordination notes for Issue #2193 -->

# Issue #2193 – Self-Test Workflow Consolidation

## Scope
- Remove legacy self-test callers (old maint/selftest wrappers that triggered PR or scheduled runs).
- Retain a single human-facing caller that triggers only on manual dispatch.
- Keep the reusable composite powering the caller consistent with the manual self-test matrix.
- Update docs/tests to reflect the new roster so regressions are caught automatically.

## Acceptance Criteria
- ✅ Only the designated manual self-test workflow appears in the Actions tab.
- ✅ The manual self-test workflow delegates to the reusable matrix via `uses: ./.github/workflows/reusable-10-ci-python.yml`.
- ✅ No other workflow names containing “selftest” remain.
- ✅ CI guardrails assert the above invariants.

## Task Checklist
- [ ] Delete the redundant self-test workflow callers.
- [ ] Ensure the manual caller exposes only `workflow_dispatch` (no cron).
- [ ] Confirm delegation to the reusable CI matrix via `./.github/workflows/reusable-10-ci-python.yml` with inherited secrets.
- [ ] Update documentation references (`docs/ci/WORKFLOWS.md`, audit sheets) if required.
- [ ] Add/extend automated tests so the roster cannot regress.
- [ ] Run workflow guardrail tests (`pytest tests/test_workflow_*.py`).

## Notes
- Keep the workflow title `Selftest 80 Reusable CI Matrix` aligned with the filename-driven guard tests.
- Avoid adding extra inputs or triggers—the matrix scenarios live inside the reusable job.
