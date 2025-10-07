# PR Gate & CI Workflow Enhancement Plan

## Scope and Key Constraints
- **Workflows affected**: `.github/workflows/gate.yml`, `.github/workflows/ci.yml`, `.github/workflows/docker.yml` (reused by gate).
- **Functional focus**: restore the gate workflow so it orchestrates the CI matrix (Python 3.11/3.12) and docker smoke tests, enforce lint/type checks before pytest, and skip heavy jobs for docs-only PRs.
- **Tooling**: GitHub Actions only; reuse existing reusable workflows instead of duplicating job steps.
- **Concurrency rules**: gate runs must cancel when superseded by newer pushes on the same PR or branch.
- **Test selection**: retain existing pytest marker expression semantics (`not quarantine and not slow`) without YAML anchors.
- **Caching requirements**: add Python setup with pip cache and ensure pytest cache persists between runs where applicable.
- **Out of scope**: changing core test coverage, altering docker workflow internals, or redefining branch protection policies beyond requiring the new gate check.

## Acceptance Criteria / Definition of Done
1. **Single Required Check**: Only one required status named `gate` appears on PRs and succeeds when all dependent jobs complete.
2. **Gate Orchestration**: `gate` job depends on three reusable-workflow jobs:
   - `core-tests-311` → `.github/workflows/ci.yml` with `python: "3.11"` and `marker: "not quarantine and not slow"`.
   - `core-tests-312` → `.github/workflows/ci.yml` with `python: "3.12"` and the same marker literal.
   - `docker-smoke`  → `.github/workflows/docker.yml`.
3. **Workflow Triggers**: `gate.yml` contains `concurrency: { group: pr-${{ github.event.pull_request.number || github.ref_name }}-gate, cancel-in-progress: true }` and ignores doc-only changes via `on.pull_request.paths-ignore` for patterns such as `**/*.md`, `docs/**`, and `assets/**`.
4. **Reusable CI Enhancements**: `.github/workflows/ci.yml` executes steps in order—Python setup with pip caching → `ruff check --output-format github` (no fix) → `mypy` → `pytest`—and fails fast if lint or type checking fails.
5. **Coverage Artifact**: Each CI run uploads coverage as `coverage-<python-version>` for the gate to consume.
6. **Caching**: Pip dependencies and pytest cache directories leverage GitHub Actions caching (via `actions/setup-python@v5` or explicit cache steps).
7. **Docs-only Short Circuit**: PRs limited to documentation/assets skip the heavy CI jobs yet still report a passing gate check.
8. **Branch Protection**: Repository settings updated so that the `gate` status is the sole required check (tracked externally but noted here).

## Initial Task Checklist
- [x] Update `.github/workflows/gate.yml`:
  - [x] Replace YAML anchor usage with inline marker strings.
  - [x] Add `concurrency` configuration and `paths-ignore` filters.
  - [x] Ensure jobs invoke reusable workflows and add a final `gate` aggregation job.
- [x] Modify `.github/workflows/ci.yml`:
  - [x] Add Python setup with caching (actions/setup-python@v5).
  - [x] Insert sequential steps for Ruff, MyPy, then Pytest with fail-fast behaviour.
  - [x] Publish coverage artifact named `coverage-<python-version>`.
- [x] Verify `.github/workflows/docker.yml` compatibility with gate reuse (no changes expected).
- [x] Coordinate with repository settings to require only the new `gate` status.
- [x] Document any additional follow-up tasks or gaps discovered during implementation.
