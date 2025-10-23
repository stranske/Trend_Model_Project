# Issue #2615 – Agents topology consolidation

_Last reviewed: 2026-10-14_

## Scope & Key Constraints
- **Single entry point**: `agents-70-orchestrator.yml` must be the only documented automation entry point for Codex-initiated or maintainer-triggered agent runs. Any other workflows should be referenced solely as supporting infrastructure.
- **Bridge continuity**: Retain `agents-63-codex-issue-bridge.yml` to translate `agent:codex`-labeled issues into branches/PRs without modifying its current trigger semantics or rate limiting.
- **Template introduction**: Add an "Agent task" issue template that pre-applies the `agents` and `agent:codex` labels and captures problem framing (background, goals, guardrails, notes) suitable for automation hand-off.
- **Compatibility cleanup**: Remove the deprecated `agents-61` and `agents-62` workflows and scrub docs/configs that might still reference them.
- **Documentation alignment**: Update `docs/ci/WORKFLOW_SYSTEM.md`, workflow READMEs, and related onboarding docs to advertise the orchestrator + bridge topology and link the new issue template.
- **Operational guardrails**: Preserve Gate and core CI expectations—no new required checks, no change in concurrency groups, and ensure existing dispatch paths keep working after the cleanup.

## Acceptance Criteria / Definition of Done
1. `.github/ISSUE_TEMPLATE/agent_task.yml` exists, automatically applies the `agents` + `agent:codex` labels, and prompts for background, goals, guardrails, and additional notes.
2. `agents-70-orchestrator.yml` is explicitly documented as the single entry point across contributor docs, with references to legacy `agents-61`/`agents-62` removed.
3. `agents-63-codex-issue-bridge.yml` remains enabled, responds to the `agent:codex` label, and is linked from documentation that explains the issue-to-PR automation flow.
4. `agents-61` and `agents-62` workflows and any related scheduler hooks/configuration are removed from the repository.
5. Documentation updates provide a clear lifecycle overview for Agent tasks (issue creation → labeling → bridge → orchestrator) with links to evidence or run instructions.
6. Required CI workflows (Gate, python ci, docker smoke) continue to pass after the changes, demonstrating no regression in automation coverage.

## Initial Task Checklist
- [x] Create `.github/ISSUE_TEMPLATE/agent_task.yml` with labels, prompts, and instructions for Codex automation requests.
- [x] Remove `.github/workflows/agents-61*.yml` and `.github/workflows/agents-62*.yml`, along with any documentation or configuration references.
- [x] Review and update `.github/workflows/README.md`, `docs/ci/WORKFLOW_SYSTEM.md`, and `docs/ops/codex-bootstrap-facts.md` to reference only the orchestrator entry point plus the codex issue bridge.
- [x] Verify `agents-63-codex-issue-bridge.yml` documentation clearly states how labeled issues trigger branch/PR creation and that no settings changes break the trigger path.
- [x] Run Gate locally or via CI to confirm required checks still succeed after workflow changes.
- [x] Capture a short verification log (issue template dry run, bridge-trigger evidence) to accompany the implementation PR.

## Verification Log
- 2026-10-14: Parsed `.github/ISSUE_TEMPLATE/agent_task.yml` to confirm the template auto-applies the `agents` and `agent:codex` labels and captures the required background and goals prompts.
- 2026-10-14: Inspected `.github/workflows/agents-63-codex-issue-bridge.yml` guards to verify `agent:codex` labels trigger bridge runs for newly created or relabeled issues.
- 2026-10-14: `pytest tests/test_workflow_agents_consolidation.py tests/test_workflow_naming.py` (pass) to demonstrate Gate-dependent workflow tests continue succeeding after the cleanup.

