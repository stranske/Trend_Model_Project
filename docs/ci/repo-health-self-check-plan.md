# Plan: Repository Health Self-Check Workflow Remediation

## Scope and Key Constraints
- **Workflow coverage:** Update `.github/workflows/repo-health-self-check.yml` (and any referenced reusable jobs) so cron and manual dispatch succeed without requiring unsupported permission scopes.
- **Permissions model:** Limit `permissions` to GitHub-supported scopes (`contents: read`, `issues: write`) and treat branch-protection inspection as best-effort when permissions are insufficient.
- **Issue lifecycle:** Ensure only a single tracker issue titled `[health] repository self-check failed` is open at any time; prefer updating an existing issue rather than creating duplicates.
- **Archival requirement:** Retire experimental self-test workflows (`maint-43-…`, `maint-44-…`, `maint-48-…`, `maint-90-…`, `pr-20-…`) by moving them under `Old/` (or removing them) and document their archival location in `ARCHIVE_WORKFLOWS.md` plus `docs/ci/WORKFLOWS.md`.
- **Documentation alignment:** Capture the new workflow intent, permission rationale, and archival notes in `docs/ci/WORKFLOWS.md` (and any linked CI documentation) without deviating from repository doc style.
- **Backward compatibility:** Preserve existing probes for labels and `SERVICE_BOT_PAT`, and keep automation that updates/creates failure issues intact.

## Acceptance Criteria / Definition of Done
1. `repo-health-self-check.yml` validates successfully (no schema errors) and executes on both scheduled (`cron`) and `workflow_dispatch` triggers.
2. Branch protection checks handle 403 responses gracefully by emitting a warning (e.g., job output `protection_issue="Unable to verify…"`) without failing the workflow.
3. When the workflow detects any failure condition, it updates or creates a single GitHub issue named `[health] repository self-check failed` with aggregated diagnostics and multi-line details.
4. Deprecated self-test workflows are removed from active CI runs, relocated/archived as required, and their new status is reflected in both `ARCHIVE_WORKFLOWS.md` and `docs/ci/WORKFLOWS.md`.
5. Documentation explains the updated workflow behaviour, permission scopes, and degradation strategy, ensuring maintainers understand the fallback logic.
6. Repository automation remains compliant with GitHub permission policies (no unsupported scopes) and produces passing CI runs post-change.

## Initial Task Checklist
- [ ] Audit the current `repo-health-self-check` workflow to catalogue permissions, branch-protection calls, and failure aggregation logic.
- [ ] Replace unsupported `permissions.administration` usage with supported granular scopes; ensure the workflow compiles locally via `act` or `workflow` schema validation.
- [ ] Update branch-protection probe to catch 403 responses, emit a warning output, and propagate the degraded state to the aggregation step without failing the job.
- [ ] Confirm the aggregation step assembles failure reasons into a single multi-line output and feeds the issue management logic.
- [ ] Refine the issue update/create step to guarantee idempotent handling of the `[health] repository self-check failed` tracker issue.
- [ ] Archive the listed self-test workflows (move under `Old/` or delete) and annotate their disposition in `ARCHIVE_WORKFLOWS.md` and `docs/ci/WORKFLOWS.md` with pointers to any surviving reusable examples.
- [ ] Review related documentation (e.g., `docs/repo_health_self_check.md`) and update as needed to reflect permission changes and fallback behaviour.
- [ ] Smoke-test the workflow via `workflow_dispatch` (or `act`) to verify cron/manual compatibility and graceful handling of restricted permissions.
