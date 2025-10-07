# PR Gate & CI Workflow Enhancement Plan# PR Gate & CI Workflow Enhancement Plan



## Scope and Key Constraints## Scope and Key Constraints

- **Workflows affected**: `.github/workflows/pr-10-ci-python.yml`, `.github/workflows/pr-12-docker-smoke.yml`, `.github/workflows/reusable-90-ci-python.yml`, and supporting maintenance listeners (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`).- **Workflows affected**: `.github/workflows/pr-10-ci-python.yml`, `.github/workflows/pr-12-docker-smoke.yml`, `.github/workflows/reusable-90-ci-python.yml`, and supporting maintenance listeners (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`).

- **Functional focus**: ensure the PR CI gate pairs (`ci / python`, `ci / docker smoke`) remain fast, deterministic, and self-contained while continuing to skip heavy jobs on docs-only changes.- **Functional focus**: ensure the PR CI gate pairs (`ci / python`, `ci / docker smoke`) remain fast, deterministic, and self-contained while continuing to skip heavy jobs on docs-only changes.

- **Tooling**: GitHub Actions only; lean on reusable composites (`reusable-90-ci-python.yml`, `reusable-92-autofix.yml`) instead of duplicating job steps.- **Tooling**: GitHub Actions only; lean on reusable composites (`reusable-90-ci-python.yml`, `reusable-92-autofix.yml`) instead of duplicating job steps.

- **Concurrency rules**: PR jobs must cancel when superseded by newer pushes on the same branch/PR.- **Concurrency rules**: PR jobs must cancel when superseded by newer pushes on the same branch/PR.

- **Test selection**: retain the existing pytest marker expression semantics (`not quarantine and not slow`) without YAML anchors inside reusable workflows.- **Test selection**: retain the existing pytest marker expression semantics (`not quarantine and not slow`) without YAML anchors inside reusable workflows.

- **Caching requirements**: keep Python setup, pip dependency caches, and pytest cache restores so reruns stay under budget.- **Caching requirements**: keep Python setup, pip dependency caches, and pytest cache restores so reruns stay under budget.

- **Out of scope**: changing core test coverage targets, altering docker workflow internals, or redefining branch protection policies beyond the two required checks.- **Out of scope**: changing core test coverage targets, altering docker workflow internals, or redefining branch protection policies beyond the two required checks.



## Acceptance Criteria / Definition of Done## Acceptance Criteria / Definition of Done

1. **Required Checks**: PRs surface exactly two required statuses—`ci / python` (from `pr-10-ci-python.yml`) and `ci / docker smoke` (from `pr-12-docker-smoke.yml`)—both green before merge.1. **Required Checks**: PRs surface exactly two required statuses—`ci / python` (from `pr-10-ci-python.yml`) and `ci / docker smoke` (from `pr-12-docker-smoke.yml`)—both green before merge.

2. **PR 10 orchestration**: `pr-10-ci-python.yml` pins formatter/type tooling, runs Ruff→Black→mypy→pytest in sequence, uploads coverage artefacts, and honours docs-only short-circuit paths.2. **PR 10 orchestration**: `pr-10-ci-python.yml` pins formatter/type tooling, runs Ruff→Black→mypy→pytest in sequence, uploads coverage artefacts, and honours docs-only short-circuit paths.

3. **PR 12 docker smoke**: `pr-12-docker-smoke.yml` builds the PR image with Buildx, runs the health probe loop, and exposes a `debug-build` toggle without leaking credentials.3. **PR 12 docker smoke**: `pr-12-docker-smoke.yml` builds the PR image with Buildx, runs the health probe loop, and exposes a `debug-build` toggle without leaking credentials.

4. **Reusable CI hygiene**: `reusable-90-ci-python.yml` remains the single source for matrix/self-test execution and is kept in sync with PR 10 tooling pins.4. **Reusable CI hygiene**: `reusable-90-ci-python.yml` remains the single source for matrix/self-test execution and is kept in sync with PR 10 tooling pins.

5. **Coverage Artefacts**: `pr-10-ci-python.yml` publishes coverage assets consumed by downstream maintenance workflows (post-CI summary, autofix, failure tracker).5. **Coverage Artefacts**: `pr-10-ci-python.yml` publishes coverage assets consumed by downstream maintenance workflows (post-CI summary, autofix, failure tracker).

6. **Docs-only Short Circuit**: Both PR workflows keep `paths-ignore` exclusions so documentation or asset-only diffs report success quickly.6. **Docs-only Short Circuit**: Both PR workflows keep `paths-ignore` exclusions so documentation or asset-only diffs report success quickly.

7. **Branch Protection**: Repository settings (tracked out-of-band) continue to require only the `ci / python` and `ci / docker smoke` checks.7. **Branch Protection**: Repository settings (tracked out-of-band) continue to require only the `ci / python` and `ci / docker smoke` checks.



## Initial Task Checklist## Initial Task Checklist

- [ ] Validate `.github/workflows/pr-10-ci-python.yml` still mirrors reusable pins, caches dependencies, and enforces coverage minimums.<<<<<<< HEAD

- [ ] Confirm `.github/workflows/pr-12-docker-smoke.yml` honours Buildx caching and exits cleanly on health check failures.- [x] Update `.github/workflows/gate.yml`:

- [ ] Keep `.github/workflows/reusable-90-ci-python.yml` aligned with PR 10 so maintenance callers inherit identical behaviour.  - [x] Replace YAML anchor usage with inline marker strings.

- [ ] Ensure maintenance listeners (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`) reference the updated workflow names.  - [x] Add `concurrency` configuration and `paths-ignore` filters.

- [ ] Document any additional follow-up tasks or gaps discovered during implementation.  - [x] Ensure jobs invoke reusable workflows and add a final `gate` aggregation job.

- [x] Modify `.github/workflows/ci.yml`:
  - [x] Add Python setup with caching (actions/setup-python@v5).
  - [x] Insert sequential steps for Ruff, MyPy, then Pytest with fail-fast behaviour.
  - [x] Publish coverage artifact named `coverage-<python-version>`.
- [x] Verify `.github/workflows/docker.yml` compatibility with gate reuse (no changes expected).
- [x] Coordinate with repository settings to require only the new `gate` status.
- [x] Document any additional follow-up tasks or gaps discovered during implementation.
=======
- [ ] Validate `.github/workflows/pr-10-ci-python.yml` still mirrors reusable pins, caches dependencies, and enforces coverage minimums.
- [ ] Confirm `.github/workflows/pr-12-docker-smoke.yml` honours Buildx caching and exits cleanly on health check failures.
- [ ] Keep `.github/workflows/reusable-90-ci-python.yml` aligned with PR 10 so maintenance callers inherit identical behaviour.
- [ ] Ensure maintenance listeners (`maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`) reference the updated workflow names.
- [ ] Document any additional follow-up tasks or gaps discovered during implementation.
>>>>>>> 48177ac5 (chore: remove legacy gate workflows)
