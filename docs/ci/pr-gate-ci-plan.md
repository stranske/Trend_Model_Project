# Gate Workflow Consolidation Plan

## Scope and Key Constraints
- **Workflows affected**: `.github/workflows/pr-gate.yml`, `.github/workflows/reusable-ci.yml`, `.github/workflows/reusable-docker.yml`, and maintenance listeners (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`).
- **Functional focus**: keep the Gate fan-out (Python 3.11, Python 3.12, Docker smoke) fast, deterministic, and observable.
- **Tooling**: GitHub Actions only; lean on `reusable-ci.yml` and `reusable-docker.yml` instead of duplicating job steps.
- **Concurrency rules**: PR jobs cancel when superseded by newer pushes on the same branch/PR.
- **Test selection**: preserve the pytest marker expression (`not quarantine and not slow`).
- **Caching requirements**: retain Python setup/pip caches and pytest cache restores to keep reruns under budget.
- **Out of scope**: changing coverage targets or adding new protected checks beyond Gate.

## Acceptance Criteria / Definition of Done
1. **Required Check**: Branch protection relies solely on the `Gate / gate` job, which aggregates the Python and Docker legs.
2. **Reusable CI**: `reusable-ci.yml` remains the single source for Python lint/type/test execution and artefact production.
3. **Reusable Docker**: `reusable-docker.yml` provides the Docker smoke test consumed by Gate and downstream automations.
4. **Maintenance listeners**: Post-CI summary, autofix, and failure tracker workflows listen for `workflow_run` events from `Gate` only.
5. **Coverage artefacts**: Gate uploads coverage bundles per Python version for downstream consumption.
6. **Docs-only short circuit**: `paths-ignore` keeps documentation-only changes from triggering heavy jobs.
7. **Documentation**: Contributor guidance and workflow catalogs describe Gate as the single enforced check and reference the reusable entry points.

## Initial Task Checklist
- [x] Point maintenance listeners at `Gate`.
- [x] Delete legacy PR wrappers and superseded reusable CI/Docker workflows.
- [x] Confirm Gate invokes `.github/workflows/reusable-ci.yml` (Python 3.11/3.12) and `.github/workflows/reusable-docker.yml`.
- [x] Refresh docs (`CONTRIBUTING.md`, `docs/ci/WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse.md`) with the new topology.
- [x] Regenerate CI signature fixtures for the Gate topology.
