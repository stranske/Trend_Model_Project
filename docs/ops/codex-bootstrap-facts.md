# Repo Ops Facts — Agent Bootstrap

_Last updated: 2026-08-16_

Trend Model Project is a consumer of the shared agent-bootstrap system in
[`stranske/Workflows`](https://github.com/stranske/Workflows). Do not restore a
local bootstrap composite action, simulation harness, or duplicate behavior
guide here.

## Source Of Truth

- Consumer entry point: `.github/workflows/agents-issue-intake.yml`
- Shared implementation: `stranske/Workflows/.github/workflows/reusable-agents-issue-bridge.yml@main`
- Shared orchestration: `stranske/Workflows/.github/workflows/reusable-70-orchestrator-*.yml@main`
- Consumer template and documentation: `stranske/Workflows/templates/consumer-repo/`

The local intake workflow and consumer documentation are sync-managed. Fix
shared behavior in Workflows first, then let the consumer-sync campaign align
this repository.

## Pull Request State

- Automation-created pull requests are opened ready for review.
- Draft state is not a dependency, staging, stack-order, or capacity control.
- Labels, PR-body lifecycle state, disabled auto-merge, required checks, and
  exact-head merge guards carry those controls.
- Before ending a run that creates or changes a pull request, verify it is open
  and GitHub reports `isDraft=false`.
- A pre-existing draft is a recovery condition: convert it to ready. Do not
  close an otherwise valid pull request merely to free an automation slot.

## Operations

1. Apply the registry-backed `agent:*` label required by the issue-intake
   workflow.
2. To run intake manually, open **Actions → Agents Issue Intake → Run
   workflow**, choose `agent_bridge`, and supply the issue number. Scheduled
   queue progression uses the local Agents 71-73 belt.
3. Diagnose shared bootstrap failures in Workflows rather than copying its
   actions or reusable workflows into this repo.
4. Keep `Agents.md` and `CONTRIBUTING.md` aligned with the ready-for-review
   invariant.

Historical bootstrap wrappers and verification notes remain available under
the repository archives. They are evidence, not active instructions.
