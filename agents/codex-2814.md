# Codex Task #2814 — Selftest Reusables Consolidation

Reference: [Issue #2814](https://github.com/stranske/Trend_Model_Project/issues/2814)

## Task Checklist
- [x] Publish a single workflow at `.github/workflows/selftest-reusable-ci.yml` that fans out to the reusable CI matrix via `jobs.<id>.uses`.
- [x] Remove the legacy self-test wrappers so the Actions inventory exposes only the consolidated entry point.
- [x] Document the scenario matrix and expected outputs in `docs/ci/SELFTESTS.md`.

## Acceptance Criteria
- ✅ Actions now list only **Selftest: Reusables**, backed by a nightly cron run and manual dispatch mode.
- ✅ The reinstated workflow produces cosmetic-safe summaries and artifacts while keeping non-code paths untouched.
- ✅ Documentation captures the matrix structure, summary output, and comment marker for auditing runs.
