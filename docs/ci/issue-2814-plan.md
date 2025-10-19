# Issue #2814 Self-Test Reusable Consolidation – Planning Notes

## Scope and Key Constraints
- **Single entry point**: Keep only one visible GitHub Actions workflow titled "Selftest: Reusables" that fans out to the reusable Python CI matrix with `jobs.<id>.uses` and preserves the six-scenario coverage.
- **Nightly and manual parity**: Retain the 06:30 UTC cron schedule while supporting `workflow_dispatch` inputs for comment mode, dual-runtime requests, history downloads, and custom reasons without regressing past automation callers.
- **Verification surface**: Continue writing a compact matrix table to `GITHUB_STEP_SUMMARY`, upload the JSON report artifact, and ensure failure guards halt the workflow when verification mismatches occur.
- **Documentation alignment**: Update CI references so contributors know where the consolidated workflow lives, which scenarios execute, and how to consume the artifacts.
- **Guardrail updates**: Refresh workflow guard tests and archive notes so enforcement matches the renamed workflow and no legacy `maint-4x` wrappers resurface.

## Acceptance Criteria / Definition of Done
1. Actions shows exactly one workflow named "Selftest: Reusables" with nightly cron (`30 6 * * *`) and manual dispatch inputs covering summary/comment/dual-runtime modes. ✅
2. The workflow aggregates results via the reusable Python matrix, writes the verification table to `GITHUB_STEP_SUMMARY`, and uploads `selftest-report.json` on both nightly and manual runs. ✅
3. Documentation and workflow guard tests reference the consolidated workflow, describe the scenario matrix, and enforce the single-entry inventory. ✅

## Task Checklist
- [x] Replace the retired runner with `.github/workflows/selftest-reusable-ci.yml`, delegating to `reusable-10-ci-python.yml`, keeping `strategy.fail-fast: false`, and emitting the verification table plus JSON report artifact.
- [x] Remove legacy `maint-4x` wrappers from the active workflow inventory and update `ARCHIVE_WORKFLOWS.md` with the Issue #2814 consolidation note and date.
- [x] Document the canonical workflow behaviour in `docs/ci/SELFTESTS.md` and refresh `docs/ci/WORKFLOWS.md` and `docs/ci/WORKFLOW_SYSTEM.md` to reference the renamed entry point.
- [x] Extend workflow guard tests (`tests/test_workflow_selftest_consolidation.py`, `tests/test_workflow_naming.py`) to pin the workflow name, triggers, matrix scenarios, and summary expectations.
- [x] Run targeted pytest modules (`pytest tests/test_workflow_selftest_consolidation.py tests/test_workflow_naming.py`) to confirm the guards reflect the new configuration.

## Completion Notes
- Nightly and manual runs continue to surface the verification table via `GITHUB_STEP_SUMMARY`; comment mode reuses the refreshed marker `<!-- selftest-reusable-comment -->` to avoid duplicates.
- The documentation stack now funnels contributors to `docs/ci/SELFTESTS.md` for scenario and artifact details, and the archive ledger records the Issue #2814 cutover.
