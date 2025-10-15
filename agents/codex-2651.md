# Issue #2651 — Self-test runner consolidation plan

## Scope & Key Constraints
- **Create a single entrypoint**: author `.github/workflows/selftest-runner.yml` that exposes the three requested inputs (`mode`, `post_to`, `enable_history`) and delegates execution to `selftest-81-reusable-ci.yml` as the matrix worker.
- **Retire wrapper scatter**: replace every existing self-test wrapper workflow with thin invocations of the new runner (or delete redundant files outright) while preserving the `selftest-81` reusable workflow untouched.
- **Preserve operational behaviour**: maintain current triggering semantics (manual `workflow_dispatch`, scheduled or reusable callers) and ensure PR comment/status publishing remains opt-in via the new inputs.
- **Guardrails**: ensure missing verification artifacts fail fast with helpful messaging, respect concurrency controls already defined for self-tests, and avoid breaking consumers that call the reusable workflow directly.
- **Documentation alignment**: update overview and maintenance docs so on-call engineers know to launch self-tests exclusively through the new runner.

## Acceptance Criteria / Definition of Done
- `.github/workflows/selftest-runner.yml` exists, is lint-clean, and can drive summary, comment, and dual-runtime modes via input parameters in both manual and reusable contexts.
- All legacy `selftest-*` wrapper workflows (except `selftest-81-reusable-ci.yml`) are either removed or replaced with calls into the new runner without behavioural regressions.
- Documentation covering self-tests (overview, reuse guides, maintenance playbooks) references the runner as the canonical entrypoint with accurate instructions.
- CI passes for the modified workflows, and an audit of the repo confirms no stray wrapper workflows remain.

## Initial Task Checklist
1. Inventory existing self-test workflows and catalogue their triggers, inputs, and consumers.
2. Design the `selftest-runner.yml` workflow structure, mapping legacy behaviours to the new input matrix and defining outputs/permissions.
3. Implement the runner workflow, including conditional comment/summary publishing and history toggling based on inputs.
4. Migrate or delete legacy wrapper workflows, ensuring any reusable-callers now invoke the runner.
5. Update relevant documentation (`ARCHIVE_WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse*.md`, etc.) to describe the consolidated runner.
6. Run workflow linting or dry-run validations as available, then open a PR with the consolidated changes and ensure Gate succeeds.
