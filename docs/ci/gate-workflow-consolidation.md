# Gate Workflow Consolidation Plan

## Scope and Key Constraints
- **Workflows in scope**: `.github/workflows/pr-00-gate.yml`, `.github/workflows/reusable-10-ci-python.yml`, `.github/workflows/reusable-12-ci-docker.yml`, and any workflow listeners that consume CI results (for example `maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`). Legacy wrappers (`pr-10-ci-python.yml`, `pr-12-docker-smoke.yml`) were removed after the migration window closed.
- **Branch protection**: Default branch must require the `Gate / gate` check name only; removal of the legacy `ci / python` and `ci / docker smoke` status checks happens in coordination with GitHub settings owners.
- **Downstream automation**: All automation that reacts to PR CI results must listen to the Gate workflow's `workflow_run` events and not depend on the deleted jobs.
- **Reusable entry points**: `reusable-10-ci-python.yml` (Python matrix) and `reusable-12-ci-docker.yml` (Docker smoke) remain the canonical job definitions; Gate orchestrates them without duplicating logic.
- **Operational safeguards**: Preserve docs-only fast paths, concurrency cancellation, environment hardening, and artifact publishing that downstream jobs consume.
- **Out of scope**: Changing test matrices, altering Docker build internals, or redefining maintenance workflows beyond updating their triggers.

## Acceptance Criteria / Definition of Done
1. Branch protection on the default branch requires only the `Gate / gate` status.
2. Legacy workflows (`pr-10-ci-python.yml`, `pr-12-docker-smoke.yml`) are removed from the repository once Gate parity is confirmed.
3. Maintenance workflows that trigger on `workflow_run` events (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`, and similar listeners) monitor `Gate` exclusively.
4. The Gate workflow fans out to `reusable-10-ci-python.yml` for Python 3.11 and 3.12 plus `reusable-12-ci-docker.yml` for the Docker smoke test, and aggregates results into a final `gate` job.
5. Documentation (`CONTRIBUTING.md`, workflow catalogs, automation guides) describes `Gate / gate` as the single required PR check and outlines the expected signals for contributors.
6. Successful Gate runs continue to publish the artefacts relied upon by downstream maintenance automation.

## Completion Checklist
- [x] Verify `.github/workflows/pr-00-gate.yml` invokes the reusable CI and Docker composites for Python 3.11, Python 3.12, and the smoke test, with results collated under the `gate` job.
- [x] Update branch protection settings to require `Gate / gate` and remove legacy required checks.
- [x] Migrate all `workflow_run` listeners to reference only the `Gate` workflow name.
- [x] Confirm Gate maintains docs-only path filters and concurrency cancellation semantics from the legacy workflows.
- [x] Delete `.github/workflows/pr-10-ci-python.yml` and `.github/workflows/pr-12-docker-smoke.yml` after validating Gate stability.
- [x] Adjust contributor documentation to reference the Gate workflow as the single enforced PR status.
- [x] Audit downstream tooling (autofix, failure tracker, post-CI summary) to ensure artefact expectations remain valid when only Gate runs.
