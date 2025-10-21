# Issue #2882 – CI Unification Plan

## Scope / Key Constraints
- [x] Unify lint, type checks, tests, and coverage packaging inside `.github/workflows/reusable-10-ci-python.yml`, keeping deterministic artifacts such as `artifacts/coverage/coverage.xml` and summary JSON outputs.
- [x] Keep `.github/workflows/pr-00-gate.yml` as the single branch-protection entry point with a minimal interpreter matrix while preserving docs-only fast-pass and Docker-change detection short circuits.
- [x] Remove duplicate coverage publishing from `.github/workflows/maint-46-post-ci.yml` and ensure scheduled coverage guard jobs auto-detect the new artifact layout.
- [x] Maintain a single `Gate / gate` status context with readable lint, type, test, and coverage summaries in the Gate job output.

## Acceptance Criteria / Definition of Done
- [x] Gate triggers one reusable Python CI job that emits lint/type/test summaries and a stable `artifacts/coverage` bundle (e.g., `coverage.xml`, `coverage.json`).
- [x] Post-CI workflows consume coverage exclusively from the Gate-produced artifact without repackaging, and scheduled coverage guard runs skip gracefully when artifacts are absent.
- [x] Docs-only changes continue to short-circuit heavy jobs, and Docker smoke executes only when relevant paths change.
- [x] Branch protection relies solely on the `Gate / gate` status, whose summary clearly surfaces lint, typing, testing, and coverage metrics.

## Task Checklist
- [x] Bundle lint, type, test, and coverage outputs under `artifacts/coverage` within `.github/workflows/reusable-10-ci-python.yml` so downstream jobs can consume a deterministic payload.
- [x] Emit structured summaries (e.g., JSON) from the reusable workflow that capture pass/fail details for linting, typing, testing, and coverage metrics.
- [x] Update `.github/workflows/pr-00-gate.yml` to invoke the reusable workflow with the minimal Python matrix required for Gate.
- [x] Render the reusable workflow summaries inside the Gate job output while keeping docs-only and Docker-change short-circuit logic intact.
- [x] Adjust `.github/workflows/maint-46-post-ci.yml` to download the Gate-produced coverage bundle instead of generating a replacement artifact.
- [x] Teach `.github/workflows/maint-coverage-guard.yml` (and related scheduled jobs) to locate the new artifact layout and skip gracefully when coverage data is unavailable.
- [x] Validate branch protection expectations and document any operational updates, including artifact paths and new safeguards or assertions added to the workflows.
