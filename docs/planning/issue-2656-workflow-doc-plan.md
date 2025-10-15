# Issue #2656 – Workflow documentation alignment plan

## Scope and key constraints
- Refresh the top-level contributor entry points (README.md and CONTRIBUTING.md) so they route users to the Workflow System Overview and the canonical workflow catalog before any legacy links.
- Update `docs/ci/WORKFLOWS.md` to ensure every referenced automation flow either maps to the current overview/canonical roster or is explicitly marked as archived, avoiding any stale CI entry points.
- Prune and redirect references to deprecated "consumer" workflows outside the optional archive record, while keeping valid gate/autofix guidance intact.
- Maintain a lightweight historical ledger in `ARCHIVE_WORKFLOWS.md` without resurrecting or altering the retired jobs.
- Keep the change set documentation-only; do not modify workflow YAML, scripts, or automation behaviour.

## Acceptance criteria / definition of done
- README.md and CONTRIBUTING.md clearly direct readers to the Workflow System Overview and to the authoritative workflow list as their first navigation step.
- `docs/ci/WORKFLOWS.md` synchronises with the overview document: active flows are confirmed, label gating and execution conditions are accurate, and no references remain to removed or renamed workflows.
- Any mention of retired workflows lives solely in `ARCHIVE_WORKFLOWS.md`, which is updated (if needed) to reflect their retired status and current routing guidance.
- Links to the overview and workflow catalog resolve correctly within the repository.
- The resulting diff is limited to documentation files and passes the docs-only gate checks.

## Initial task checklist
1. Review the latest Workflow System Overview and canonical workflow roster to capture the authoritative terminology and URLs.
2. Audit README.md and CONTRIBUTING.md for existing workflow references; update intro sections to point to the overview/canonical roster and remove stale links.
3. Walk through `docs/ci/WORKFLOWS.md`, trimming or revising entries so only active flows remain, and ensure notes about label-gated or optional jobs align with current behaviour.
4. Inspect `ARCHIVE_WORKFLOWS.md`; append or adjust entries to confirm retired workflows are still deprecated and that readers are redirected to the overview for the active list.
5. Run or verify docs-only CI checks (gate fast-pass) and prepare the PR summary referencing the documentation updates.
