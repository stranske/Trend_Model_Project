# Issue #2650 — Transition to Orchestrator + Issue Bridge

## Scope & Key Constraints
- Remove the legacy consumer workflow (`.github/workflows/agents-62-consumer.yml`) and any references to it without touching unrelated automation.
- Keep `.github/workflows/agents-70-orchestrator.yml` as the single entry point for agent execution; ensure any updates preserve its parameters and downstream reusable jobs.
- Verify `.github/workflows/agents-63-codex-issue-bridge.yml` continues to target the orchestrator and only triggers on `agents` + `agent:codex` labelled issues.
- Confine issue-template edits to applying the correct labels and guidance; avoid restructuring other templates or global configuration.
- Maintain Gate, actionlint, and workflow validation compatibility while making the changes.

## Acceptance Criteria / Definition of Done
1. `.github/workflows/agents-62-consumer.yml` is deleted (and archived if required) so no consumer flow remains.
2. Orchestrator and issue bridge documentation reflects the orchestrator-only path, including trigger labels and hand-off behaviour.
3. The “Agent task” issue template automatically applies the `agents` and `agent:codex` labels and instructs authors on bridge expectations.
4. Creating an issue with the updated template triggers the issue bridge, resulting in a working branch/PR aligned with the orchestrator workflow.
5. Required CI (Gate, workflow linting/tests) passes after the workflow and template updates.

## Initial Task Checklist
- [ ] Delete `.github/workflows/agents-62-consumer.yml` and update archival notes (`ARCHIVE_WORKFLOWS.md` or similar) if policy requires historical tracking.
- [ ] Audit other workflows, docs, and scripts for references to the consumer path; replace them with orchestrator-only guidance.
- [ ] Confirm `.github/workflows/agents-63-codex-issue-bridge.yml` hands off exclusively to `agents-70-orchestrator.yml`, updating triggers or inputs as needed.
- [ ] Update `.github/ISSUE_TEMPLATE/agent_task.yml` so it prelabels issues with `agents` and `agent:codex` (or confirm the existing automation already does so) and clarifies orchestrator expectations.
- [ ] Dry-run or document an issue-bridge execution that creates a branch/PR via the orchestrator, capturing run links for validation.
- [ ] Run workflow validation checks (e.g., `make gate`, `npm run actionlint`, or targeted pytest) to ensure CI compliance post-change.
