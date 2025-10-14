# Maint-47 Shim Removal Plan

## Scope and Key Constraints
- Remove the legacy compatibility workflows:
  - `.github/workflows/maint-47-check-failure-tracker.yml`
  - `.github/workflows/agents-61-consumer-compat.yml`
  - `.github/workflows/agents-62-consumer.yml`
- Update documentation (`docs/ci/WORKFLOWS.md` and `docs/ci/WORKFLOW_SYSTEM.md`) to eliminate references to the removed workflows and clarify the Maint-46/orchestrator flow.
- Maintain compatibility with the current orchestrator-based CI pipeline; no changes to active Maint-46 workflows outside of reference updates.
- Preserve changelog and historical audit trails (e.g., `ARCHIVE_WORKFLOWS.md`) by not altering archived records beyond necessary cross-links.
- Respect existing documentation structure and formatting conventions in the CI docs section.

## Acceptance Criteria / Definition of Done
1. The three legacy workflow files are deleted from `.github/workflows` and no longer referenced anywhere in the repository.
2. `docs/ci/WORKFLOWS.md` and `docs/ci/WORKFLOW_SYSTEM.md` reflect the updated workflow inventory, with no lingering mentions of Maint-47 shims.
3. CI (Gate and downstream maintained suites) runs cleanly after the removals using the existing automation.
4. All references in supporting docs or scripts (search-driven) either remain valid or are updated to point to the Maint-46/orchestrator equivalents.
5. PR includes confirmation of search for residual references to the removed workflows.

## Initial Task Checklist
- [x] Delete the three legacy workflow YAML files under `.github/workflows`.
- [x] Run a repository-wide search for `maint-47`, `agents-61-consumer-compat`, and `agents-62-consumer` to identify documentation or script references.
- [x] Update `docs/ci/WORKFLOWS.md` to remove the obsolete entries and, if necessary, note the Maint-46/orchestrator replacements.
- [x] Update `docs/ci/WORKFLOW_SYSTEM.md` to align the system overview with the current workflow set.
- [x] Verify no other docs (e.g., `ARCHIVE_WORKFLOWS.md`, README snippets) need adjustment based on the search results.
- [x] Run the Gate workflow (or rely on CI automation) to confirm the pipeline remains green.

Gate run 18506446666 completed successfully after the removals, and the search above confirmed only archival references remain.
