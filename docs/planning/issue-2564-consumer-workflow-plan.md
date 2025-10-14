# Issue #2564 — Agents 61/62 Consumer Workflow Decision Plan

## Scope and Key Constraints
- Assess the current `Agents 61 Consumer Compat` and `Agents 62 Consumer` GitHub workflows to determine whether they should remain as manual-only compatibility shims or be archived entirely.
- Limit changes to the consumer workflow definitions, their related documentation (e.g., `Agents.md`, `ARCHIVE_WORKFLOWS.md`), and any supporting runbooks or audit files directly impacted by the decision.
- Preserve the Orchestrator workflow as the canonical automated entry point; no new automation may be introduced for the consumer workflows.
- Ensure any retained workflow is safe for manual invocation by adding appropriate concurrency controls (`agents-consumer-${{ github.ref }}`) and by clearly labeling the workflow intent.
- If archiving, follow the repository’s established archival pattern (move to `Old/` or remove with documentation) without disrupting historical references or automation relying on file presence.

## Acceptance Criteria / Definition of Done
- A deliberate decision is documented describing whether the consumer workflows are retained as manual shims or archived.
- For the “retain” path: both workflows trigger exclusively via `workflow_dispatch`, include the shared concurrency guard, and contain clear nomenclature noting their compatibility/shim status.
- For the “archive” path: workflows are relocated (e.g., to `Old/`) or deleted, with `ARCHIVE_WORKFLOWS.md` capturing the rationale, and no automated triggers remain.
- `Agents.md` (or successor documentation) is updated to reflect the final status and to reaffirm the Orchestrator as the canonical entry point.
- CI passes (Gate + any affected validations) after the modifications, and documentation builds/links remain valid.

## Initial Task Checklist
1. Inventory current triggers, concurrency settings, and documentation for the Agents 61/62 consumer workflows.
2. Consult stakeholders or historical context to choose between “manual shim” vs “archive,” noting dependencies.
3. Implement the selected path:
   - **If retain:** strip non-manual triggers, add concurrency guard, adjust naming/description, and ensure manual invocation documentation is accurate.
   - **If archive:** relocate or remove workflow files and update `ARCHIVE_WORKFLOWS.md` (or equivalent) with rationale.
4. Update `Agents.md` (and any relevant runbooks) with the decision outcome and guidance for future use.
5. Run Gate (and any necessary supplemental checks) to confirm CI health.
6. Solicit review, capture approvals, and merge once acceptance criteria are satisfied.
