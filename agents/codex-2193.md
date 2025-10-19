<!-- Coordination notes for Issue #2193 -->

# Issue #2193 – Self-Test Workflow Consolidation

> _Update 2026-11-04:_ Subsequent consolidation (Issue #2651) replaced the `maint-90`/`reusable-99` pair with `selftest-81-reusable-ci.yml` plus the `selftest-reusable-ci.yml` entry point. _Update 2027-02:_ The reusable matrix now lives directly in `selftest-reusable-ci.yml`; references to `selftest-81-reusable-ci.yml` are historical.

## Scope
- Remove legacy self-test callers (`selftest-82-pr-comment.yml`, `maint-43-*.yml`, `maint-44-*.yml`, `maint-48-*.yml`, `selftest-80-pr-comment.yml`).
- Retain a single human-facing caller (`maint-90-selftest.yml`) that triggers only on manual dispatch and the scheduled sweep.
- Keep `reusable-99-selftest.yml` as the sole reusable composite powering the caller.
- Update docs/tests to reflect the new roster so regressions are caught automatically.

## Acceptance Criteria
- ✅ Only `maint-90-selftest.yml` appears in the Actions tab for self-tests.
- ✅ `maint-90-selftest.yml` delegates to `reusable-99-selftest.yml` via `uses: ./.github/workflows/reusable-99-selftest.yml`.
- ✅ No other workflow names containing “selftest” remain.
- ✅ CI guardrails assert the above invariants.

## Task Checklist
- [ ] Delete the redundant self-test workflow callers.
- [ ] Ensure the new caller is workflow-dispatch + weekly cron only.
- [ ] Confirm delegation to `reusable-99-selftest.yml` with inherited secrets.
- [ ] Update documentation references (`docs/ci/WORKFLOWS.md`, audit sheets) if required.
- [ ] Add/extend automated tests so the roster cannot regress.
- [ ] Run workflow guardrail tests (`pytest tests/test_workflow_*.py`).

## Notes
- Post-CI summary and failure tracker workflows expect the caller to be named `Maint 90 Selftest`; avoid renaming without updating their allowlists.
- Keep the weekly cron lightweight—no matrix overrides or additional inputs should be added at the caller level.
