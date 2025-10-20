# Issue #2882 – CI Unification Plan

## Scope / Key Constraints
- [ ] Unify lint, type checks, tests, and coverage packaging inside `.github/workflows/reusable-10-ci-python.yml`, keeping deterministic artifacts such as `artifacts/coverage/coverage.xml` and summary JSON outputs.
- [ ] Keep `.github/workflows/pr-00-gate.yml` as the single branch-protection entry point with a minimal interpreter matrix while preserving docs-only fast-pass and Docker-change detection short circuits.
- [ ] Remove duplicate coverage publishing from `.github/workflows/maint-46-post-ci.yml` and ensure scheduled coverage guard jobs auto-detect the new artifact layout.
- [ ] Maintain a single `Gate / gate` status context with readable lint, type, test, and coverage summaries in the Gate job output.

## Acceptance Criteria / Definition of Done
- [ ] Gate triggers one reusable Python CI job that emits lint/type/test summaries and a stable `artifacts/coverage` bundle (e.g., `coverage.xml`, `coverage.json`).
- [ ] Post-CI workflows consume coverage exclusively from the Gate-produced artifact without repackaging, and scheduled coverage guard runs skip gracefully when artifacts are absent.
- [ ] Docs-only changes continue to short-circuit heavy jobs, and Docker smoke executes only when relevant paths change.
- [ ] Branch protection relies solely on the `Gate / gate` status, whose summary clearly surfaces lint, typing, testing, and coverage metrics.

## Task Checklist
- [ ] Normalize reusable Python CI outputs by editing `.github/workflows/reusable-10-ci-python.yml` to bundle lint/type/test results with coverage under `artifacts/coverage`, generate summary JSON, and upload a deterministically named artifact.
- [ ] Integrate reusable outputs into Gate by updating `.github/workflows/pr-00-gate.yml` to call the workflow with the minimal matrix, parse the summaries for the job output, and preserve docs-only and Docker-only short-circuit paths.
- [ ] Streamline Post-CI coverage handling by adjusting `.github/workflows/maint-46-post-ci.yml` to download the Gate artifact and teaching `.github/workflows/maint-coverage-guard.yml` (and related jobs) to locate or gracefully skip when the bundle is missing.
- [ ] Confirm branch protection setup so that only `Gate / gate` is required, updating operational documentation if needed to describe the coverage artifact path and workflow changes.
- [ ] Add optional safeguards by incorporating workflow assertions (e.g., via `actions/github-script`) to fail early on missing artifacts or summary parsing errors and documenting new validation steps for CI maintenance.
