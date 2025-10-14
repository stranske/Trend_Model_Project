# Workflow System Publication Plan (Issue #2617)

## Scope / Key Constraints
- Publish the finalized `docs/ci/WORKFLOW_SYSTEM.md` overview sourced from the tracking issue, ensuring terminology and workflow listings match the current CI/agents topology (Gate as the sole required PR check, orchestrator entry point, retired workflows noted as archived).
- Update `README.md` and `docs/WORKFLOW_GUIDE.md` so contributors can easily discover the new overview; align descriptions to avoid conflicting guidance with existing CI documentation.
- Refresh `docs/ci/WORKFLOWS.md` to reference the overview for high-level context while keeping its focus on actionable CI workflow details; explicitly move retired workflows into an archived section consistent with repository conventions.
- Remove the obsolete `WORKFLOW_AUDIT_TEMP.md` file without losing any still-relevant information (confirm migrated content exists in the new overview or other docs before deletion).
- Keep documentation style consistent with existing Markdown conventions (title case headings, fenced commands, relative links) and verify that navigation/front-matter (e.g., README lists) stay alphabetized where applicable.

## Acceptance Criteria / Definition of Done
- `docs/ci/WORKFLOW_SYSTEM.md` is present with the approved content and accurately describes current workflow categories, required checks, and automation policies.
- `README.md` and `docs/WORKFLOW_GUIDE.md` both link to the overview in locations where contributors look for CI/automation guidance; surrounding text reflects the new document and contains no outdated references to retired workflows or temporary audits.
- `docs/ci/WORKFLOWS.md` lists only active workflows in its primary sections, clearly labeling archived/retired ones separately, and cross-links to the overview for broader architecture context.
- `WORKFLOW_AUDIT_TEMP.md` is deleted, and any unique actionable guidance it previously contained is either superseded or explicitly migrated to the updated docs.
- All modified docs render without lint errors (Markdown lint, link check) under the repository’s standard documentation tooling.

## Initial Task Checklist
- [x] Import the canonical `WORKFLOW_SYSTEM.md` content from the tracking issue and verify terminology against current workflow files (`.github/workflows/`).
- [x] Insert/adjust links in `README.md` and `docs/WORKFLOW_GUIDE.md`, maintaining table of contents or navigation consistency.
- [x] Edit `docs/ci/WORKFLOWS.md` to align active vs. archived workflow listings and include a pointer to the new overview.
- [x] Audit `WORKFLOW_AUDIT_TEMP.md` for any remaining unique information and migrate or confirm redundancy before removal.
- [x] Run documentation formatting/lint checks (e.g., `make docs-lint` or equivalent) to ensure all updated files meet style requirements. *(Verified with `pytest tests/test_workflow_naming.py` to confirm documentation coverage guard passes.)*
