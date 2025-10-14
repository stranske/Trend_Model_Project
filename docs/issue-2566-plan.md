# Issue #2566 — Verification Workflow Plan

## Scope and Key Constraints
- Cover the reusable verification workflow and the orchestrator wiring that calls it, focusing on inputs/outputs rather than the broader agent pipeline. 【F:.github/workflows/agents-64-verify-agent-assignment.yml†L1-L74】【F:.github/workflows/agents-70-orchestrator.yml†L1-L119】
- Preserve the existing workflow_call surface (issue number + optional valid_assignees) so downstream jobs and manual dispatches remain compatible. 【F:.github/workflows/agents-64-verify-agent-assignment.yml†L4-L33】
- Respect the resolved parameter defaults produced by the orchestrator, including the accepted assignee logins (`copilot`, `chatgpt-codex-connector`, `stranske-automation-bot`). 【F:.github/workflows/agents-70-orchestrator.yml†L28-L83】
- Keep verification logic aligned with the documented pass condition that requires the `agent:codex` label and one approved assignee. 【F:Agents.md†L140-L157】

## Acceptance Criteria / Definition of Done
- The orchestrator successfully relays `enable_verify_issue`, `verify_issue_number`, and `verify_issue_valid_assignees` into the reusable job and logs the resolved outputs. 【F:.github/workflows/agents-70-orchestrator.yml†L104-L135】
- A manual orchestrator run using `params_json` with `{ "enable_verify_issue": "true", "verify_issue_number": "<issue>" }` completes green and records the verification summary with the matched assignee in the run log. 【F:.github/workflows/agents-70-orchestrator.yml†L252-L317】
- The verify workflow treats `stranske-automation-bot` as a valid assignee when the label check passes, per the documented pass condition. 【F:.github/workflows/agents-64-verify-agent-assignment.yml†L6-L73】【F:Agents.md†L148-L155】
- Agents playbook clearly states the pass criteria and surfaced outputs so operators know when verification succeeds. 【F:Agents.md†L140-L157】

## Initial Task Checklist
- [x] Audit `.github/workflows/agents-64-verify-agent-assignment.yml` for `workflow_call` input names/types and confirm they match orchestrator expectations. 【F:.github/workflows/agents-64-verify-agent-assignment.yml†L4-L74】
- [x] Trace `.github/workflows/agents-70-orchestrator.yml` parameter resolution to ensure `verify_issue_valid_assignees` always includes the automation bot and can be overridden via `params_json`. 【F:.github/workflows/agents-70-orchestrator.yml†L28-L119】
- [x] Validate that the orchestrator job graph invokes the verify workflow and appends the run summary when verification succeeds. 【F:.github/workflows/agents-70-orchestrator.yml†L240-L317】
- [x] Prepare or identify a controlled issue for manual verification, then document the successful run link and resulting summary output. 【F:docs/evidence/agents-orchestrator/manual-run-issue-2566.md†L1-L38】
- [x] Update `Agents.md` if additional operator guidance emerges during implementation. 【F:Agents.md†L140-L157】
