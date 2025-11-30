# Issue #2651 — Self-test runner consolidation plan

## Scope & Key Constraints
- **Create a single entrypoint**: author `.github/workflows/selftest-reusable-ci.yml` that exposes the requested inputs (`mode`, `post_to`, `enable_history`) and runs the reusable CI matrix directly.
- **Retire wrapper scatter**: replace every existing self-test wrapper workflow with thin invocations of the new runner (or delete redundant files outright) while migrating the matrix logic into `selftest-reusable-ci.yml`.
- **Preserve operational behaviour**: maintain current triggering semantics (manual `workflow_dispatch`, scheduled or reusable callers) and ensure PR comment/status publishing remains opt-in via the new inputs.
- **Guardrails**: ensure missing verification artifacts fail fast with helpful messaging, respect concurrency controls already defined for self-tests, and avoid breaking consumers that call the reusable workflow directly.
- **Documentation alignment**: update overview and maintenance docs so on-call engineers know to launch self-tests exclusively through the new runner.

## Acceptance Criteria / Definition of Done
- `.github/workflows/selftest-reusable-ci.yml` exists, is lint-clean, and can drive summary, comment, and dual-runtime modes via input parameters in both manual and reusable contexts.
- All legacy `selftest-*` wrapper workflows are either removed or replaced with calls into the new runner without behavioural regressions.
- Documentation covering self-tests (overview, reuse guides, maintenance playbooks) references the runner as the canonical entrypoint with accurate instructions.
- CI passes for the modified workflows, and an audit of the repo confirms no stray wrapper workflows remain.

## Initial Task Checklist
1. Inventory existing self-test workflows and catalogue their triggers, inputs, and consumers.
2. Design the `selftest-reusable-ci.yml` workflow structure, mapping legacy behaviours to the new input matrix and defining outputs/permissions.
3. Implement the runner workflow, including conditional comment/summary publishing and history toggling based on inputs.
4. Migrate or delete legacy wrapper workflows, ensuring any reusable-callers now invoke the runner.
5. Update relevant documentation (`ARCHIVE_WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse*.md`, etc.) to describe the consolidated runner.
6. Run workflow linting or dry-run validations as available, then open a PR with the consolidated changes and ensure Gate succeeds.

## Completion Status (2026-11-04)
- [x] Consolidated the manual runner into `.github/workflows/selftest-reusable-ci.yml` with summary, comment, and dual-runtime inputs.
- [x] Deleted the legacy `selftest-8X-*` wrappers so only the consolidated `selftest-reusable-ci.yml` remains.
- [x] Updated overview, reuse, and maintenance docs to reflect the single entry point and archived the historical wrappers.
- [x] Added guardrail tests in `tests/test_workflow_selftest_consolidation.py` to lock the workflow inventory, inputs, permissions, and publish job behaviour.
