# Agent Automation & Telemetry Overview

This repository is a consumer of the shared automation in
[`stranske/Workflows`](https://github.com/stranske/Workflows). The local files
under `.github/workflows/` are deployed entry points and event routers; they are
not a second implementation of the shared orchestration logic.

## Current flow

```text
agent:* label/manual intake -> Agents Issue Intake -> shared issue bridge

agents:auto-pilot/manual Auto-Pilot -> Agents 71 -> Agents 72 dispatch wrapper
                                           |
                                           v
                               ready-for-review PR -> Gate -> Agents 81
```

- `agents-issue-intake.yml` is the canonical local front door. It reacts to
  assignment-shaped agent labels, excludes metadata-only labels, and supports
  manual `agent_bridge` dispatch.
- `agents-auto-pilot.yml` is the end-to-end label/manual route. It invokes the
  dispatchable Agents 71 queue selector and the Agents 72 dispatch wrapper.
- `agents-72-codex-belt-worker.yml` is callable-only and is invoked by
  `agents-72-codex-belt-worker-dispatch.yml` to create or refresh the PR.
- `agents-73-codex-belt-conveyor.yml` is callable-only and currently has no
  local caller. It is not an Actions entry point or the active delivery route.
- `agents-80-pr-event-hub.yml` and `agents-81-gate-followups.yml` route PR and
  Gate events; Agents 81 owns the active guarded post-Gate merge sweep.
- `agents-keepalive-sweep.yml` periodically re-evaluates stalled non-draft agent
  PRs.

The retired consumer-side orchestrator and local bootstrap simulation are not
part of this topology. Do not recreate local wrappers or expose a draft-PR
control. Automation-created and reused PRs must be ready for review; lifecycle
holds belong in labels, checks, and exact-head metadata.

## Manual intake

1. Open **Actions -> Agents Issue Intake -> Run workflow**.
2. Choose `agent_bridge`, provide the issue number, and keep the configured
   agent selection unless the issue has an explicit route.
3. Review the intake summary for the created ready-for-review PR. For an
   `agents:auto-pilot` issue, also inspect Auto-Pilot, Agents 71, and the Agents
   72 dispatch-wrapper runs.
4. Confirm Gate and all review threads against the unchanged head before merge.

Equivalent CLI dispatch:

```bash
gh workflow run agents-issue-intake.yml \
  --field mode=agent_bridge \
  --field issue_number=NUMBER
```

## Ownership and changes

Use this source-of-truth order: `stranske/Workflows` root documentation,
Workflows integration guides, consumer sync sources and templates, then local
repository-specific files.

For shared behavior, repair the canonical workflow or consumer template in
`stranske/Workflows`, validate it there, and deliver it through the managed
sync path. A local forward fix is appropriate only when it matches the prepared
source repair and closes an immediate safety gap.

Use the local workflow inventory in [`.github/workflows/README.md`](../.github/workflows/README.md)
and the detailed [workflow guide](WORKFLOW_GUIDE.md) for repository-specific
entry points. Keepalive changes must also follow
[`docs/keepalive/GoalsAndPlumbing.md`](keepalive/GoalsAndPlumbing.md).

## Authentication and telemetry

- Shared write paths prefer the repository-scoped Workflows GitHub App token;
  configured PATs and `GITHUB_TOKEN` are used only where the shared contract
  allows them.
- Intake, belt, event-hub, keepalive, and Gate run summaries are the operational
  telemetry. Investigate those runs instead of relying on retired simulation
  scripts or old workflow names.
- Authentication changes must preserve GitHub App installation checks as well
  as repository secret checks.
