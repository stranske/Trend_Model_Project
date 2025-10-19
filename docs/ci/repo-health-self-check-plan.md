# Plan: Repository Health Self-Check Workflow Remediation

## Scope and Key Constraints
- **Workflow coverage:** Update `.github/workflows/health-40-repo-selfcheck.yml` (and any referenced reusable jobs) so cron and manual dispatch succeed without requiring unsupported permission scopes.
- **Permissions model:** Limit `permissions` to GitHub-supported scopes (`contents: read`, `issues: write`, `actions: write`) and treat branch-protection inspection as best-effort when permissions are insufficient.
- **Issue lifecycle:** Ensure only a single tracker issue titled `[health] repository self-check failed` is open at any time; prefer updating an existing issue rather than creating duplicates.
- **Archival requirement:** Earlier remediation retired experimental self-test workflows (`selftest-83-…`, `selftest-84-…`, `selftest-88-…`, `maint-90-…`, `selftest-82-…`). Issue #2525 subsequently reinstated these scenarios under the numbered `selftest-8X-*` manual wrappers; Issue #2651 later replaced that roster with the single `selftest-reusable-ci.yml` entry point while keeping archival notes in `ARCHIVE_WORKFLOWS.md` and `docs/ci/WORKFLOWS.md`.
- **Documentation alignment:** Capture the new workflow intent, permission rationale, and archival notes in `docs/ci/WORKFLOWS.md` (and any linked CI documentation) without deviating from repository doc style.
- **Backward compatibility:** Preserve existing probes for labels and `SERVICE_BOT_PAT`, and keep automation that updates/creates failure issues intact.

## Acceptance Criteria / Definition of Done
1. `health-40-repo-selfcheck.yml` validates successfully (no schema errors) and executes on both scheduled (`cron`) and `workflow_dispatch` triggers.
2. Branch protection checks handle 403 and 429 responses gracefully by emitting a warning (e.g., job output `protection_issue="Unable to verify…"`) without failing the workflow.
3. When the workflow detects any failure condition, it updates or creates a single GitHub issue named `[health] repository self-check failed` with aggregated diagnostics and multi-line details.
4. Deprecated self-test workflows are removed from active CI runs, relocated/archived as required, and their new status is reflected in both `ARCHIVE_WORKFLOWS.md` and `docs/ci/WORKFLOWS.md`.
5. Documentation explains the updated workflow behaviour, permission scopes, and degradation strategy, ensuring maintainers understand the fallback logic.
6. Repository automation remains compliant with GitHub permission policies (no unsupported scopes) and produces passing CI runs post-change.

### 2026-02-18 status update
- `.github/workflows/health-40-repo-selfcheck.yml` now relies solely on the default token (`contents: read`, `issues: write`, `actions: write`) and uses `repos.getBranch` to probe protection, treating 403 and 429 responses as warnings while still flagging missing protection as errors.
- The aggregation step emits both a PR-ready checklist and a tracker body; the workflow opens or updates `[health] repository self-check failed` when warnings/errors persist and automatically closes the tracker when the run clears.
- `docs/ci/WORKFLOWS.md` documents the reduced permission surface and the tracker issue behaviour so operators know what to expect from the updated run summaries.

## Initial Task Checklist
- [x] Audit the current `health-40-repo-selfcheck` workflow to catalogue permissions, branch-protection calls, and failure aggregation logic.
- [x] Replace unsupported `permissions.administration` usage with supported granular scopes; ensure the workflow compiles locally via `workflow_dispatch` dry runs.
- [x] Update branch-protection probe to catch 403 responses, emit a warning output, and propagate the degraded state to the aggregation step without failing the job.
- [x] Confirm the aggregation step assembles failure reasons into a single multi-line output and feeds the issue management logic.
- [x] Refine the issue update/create step to guarantee idempotent handling of the `[health] repository self-check failed` tracker issue.
- [x] Archive the listed self-test workflows (move under `Old/` or delete) and annotate their disposition in `ARCHIVE_WORKFLOWS.md` and `docs/ci/WORKFLOWS.md` with pointers to any surviving reusable examples.
- [x] Review related documentation (e.g., `docs/repo_health_self_check.md`) and update as needed to reflect permission changes and fallback behaviour.
- [x] Smoke-test the workflow via `workflow_dispatch` review and guard tests to verify cron/manual compatibility and graceful handling of restricted permissions.
