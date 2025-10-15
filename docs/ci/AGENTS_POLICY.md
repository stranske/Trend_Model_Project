# Agents Workflow Protection Policy

## Scope and Purpose
- **Agents 63 Codex Issue Bridge (`.github/workflows/agents-63-codex-issue-bridge.yml`)** funnels labelled `agent:codex` issues into automation by opening or updating the coordinating pull requests.
- **Agents 63 ChatGPT Issue Sync (`.github/workflows/agents-63-chatgpt-issue-sync.yml`)** turns curated topic lists into ready-to-use GitHub issues so the bridge stays fed.
- **Agents 70 Orchestrator (`.github/workflows/agents-70-orchestrator.yml`)** is the single dispatch surface for Codex automation and must remain the only consumer entry point.

Together these workflows keep the Codex issue lifecycle healthy. If any go missing or stop matching the documented contract, agent dispatch halts.

## "Unremovable" Guard Rails
The repository enforces a layered protection model so the files above cannot be deleted, renamed, or quietly rewritten:
1. **CODEOWNERS** requires maintainer review on every change to the protected workflows.
2. **Repository ruleset** blocks deletion and rename operations for the files and restricts bypasses to maintainers.
3. **Agents Critical Guard CI** fails pull requests that break the policy and is marked as a required status.

Treat these controls as non-negotiable. If an emergency requires temporarily relaxing them, follow the override steps in the root [`docs/AGENTS_POLICY.md`](../AGENTS_POLICY.md) and restore all blocks immediately after merging.

## Allowlisted Change Reasons
Only make direct edits when one of the following is true:
- **Incident recovery** – repairing the workflow after a production failure that blocks agent dispatch.
- **Security response** – patching a vulnerability affecting workflow secrets or execution.
- **Coordinated upgrade** – updating inputs, outputs, or dependencies as part of an approved maintenance plan.

Every change must be labelled with `agents:allow-change` so reviewers and the guardrail workflows can trace the approval trail. The label is applied by a maintainer after confirming the justification. Remove it (or replace with a fresh approval) once the PR merges.

## Troubleshooting Cheatsheet
- **Agents Critical Guard fails** – inspect the workflow summary for the violating path, confirm the `agents:allow-change` label is present, and verify a maintainer approved the change.
- **Bridge stops creating PRs** – run the workflow manually with `test_issue` to validate credentials, then inspect the run logs for API errors.
- **ChatGPT sync stalls** – dispatch the workflow with the `debug` flag set to `true` and confirm the topic source (repo file, raw input, or URL) resolves correctly.
- **Orchestrator never runs** – ensure incoming issues retain the `agent:codex` label and that no other workflow consumed or closed them prematurely.
