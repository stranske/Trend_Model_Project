# Plan: Consolidate Pull-Request Checks into Gate

Issue #2439 migrates the repository from the historical `pr-10-ci-python` / `pr-12-docker-smoke` pair to a single Gate workflow
that fans out to reusable Python and Docker composites. This document captures the scope, constraints, and verification checklist
for the consolidation.

## Scope and Key Constraints

- **Workflows affected**: `.github/workflows/pr-00-gate.yml`, `.github/workflows/reusable-10-ci-python.yml`,
  `.github/workflows/reusable-12-ci-docker.yml`, and the maintenance listeners that react to CI completion
   (primarily `maint-46-post-ci.yml`).
- **Functional focus**: keep the combined Gate run fast and deterministic while preserving the docs-only short circuit.
- **Tooling**: continue using reusable workflows for the Python (3.11/3.12) and Docker jobs; the Gate file is a thin orchestrator.
- **Concurrency rules**: Gate must cancel superseded runs for the same PR branch.
- **Out of scope**: changing pytest marker expressions, altering Docker smoke internals, or redefining coverage targets.

## Acceptance Criteria / Definition of Done

1. Branch protection requires the `Gate / gate` check name (legacy `ci / python` and `ci / docker smoke` checks are removed).
2. `.github/workflows/pr-10-ci-python.yml` and `.github/workflows/pr-12-docker-smoke.yml` are deleted once Gate parity is verified.
3. Maintenance followers (`maint-46-post-ci.yml` and similar listeners) subscribe to
   Gate via `workflow_run` triggers and operate on the Gate job manifest.
4. Gate fans out to Python 3.11, Python 3.12, and Docker smoke by delegating to `reusable-10-ci-python.yml` and `reusable-12-ci-docker.yml`.
5. CI signature fixtures (`.github/signature-fixtures/*`) represent the Gate topology.
6. Documentation (CONTRIBUTING, workflow catalog, automation references) calls out Gate as the sole required PR check.

## Completion Checklist

- [x] Update branch protection to require `Gate / gate`.
- [x] Delete `.github/workflows/pr-10-ci-python.yml` and `.github/workflows/pr-12-docker-smoke.yml`.
- [x] Point `maint-46-post-ci.yml` at Gate.
- [x] Regenerate `.github/signature-fixtures/basic_jobs.json` and the corresponding hash for the Gate manifest.
- [x] Verify Gate reuses `reusable-10-ci-python.yml` for Python 3.11/3.12 and `reusable-12-ci-docker.yml` for Docker smoke.
- [x] Refresh documentation to describe Gate as the sole required check and record the archival of the legacy wrappers.
