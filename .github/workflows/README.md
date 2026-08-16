# Workflow Quick Start

This repository consumes shared agent automation from
[`stranske/Workflows`](https://github.com/stranske/Workflows). Treat the files
in this directory as the deployed consumer surface, not as a second source of
truth for shared behavior.

## Primary Entry Points

| Purpose | Workflow |
|---|---|
| Required pull-request gate | `pr-00-gate.yml` |
| Agent issue intake and manual bootstrap | `agents-issue-intake.yml` |
| End-to-end issue automation | `agents-auto-pilot.yml` |
| Codex queue claim and recovery dispatch | `agents-71-codex-belt-dispatcher.yml` |
| Manual/API belt-worker dispatch | `agents-72-codex-belt-worker-dispatch.yml` |
| PR event routing | `agents-80-pr-event-hub.yml` |
| Gate follow-up and guarded delivery | `agents-81-gate-followups.yml` |
| Stalled-PR recovery | `agents-keepalive-sweep.yml` |
| Protected-workflow enforcement | `agents-guard.yml` |

The old consumer orchestrator wrapper is retired. Shared orchestration lives in
Workflows reusable workflows; do not recreate a local wrapper.

`agents-72-codex-belt-worker.yml` and
`agents-73-codex-belt-conveyor.yml` are reusable `workflow_call` components,
not operator entry points. The worker is invoked by the 72 dispatch wrapper.
The 73 conveyor currently has no local `uses:` caller; do not describe it as an
automatic consumer route or try to dispatch it from the Actions UI. The active
post-Gate guarded merge path is in Agents 81. Any change to this source-managed
wiring must begin in `stranske/Workflows`.

## Manual Agent Intake

1. Apply the registry-backed `agent:codex` label to the issue, or open
   **Actions -> Agents Issue Intake -> Run workflow**.
2. For manual dispatch, choose `agent_bridge` and supply the issue number.
3. Leave the bridge agent as `codex` unless the issue is explicitly routed to
   another registered agent.

For the end-to-end format → agent → PR monitoring route, apply
`agents:auto-pilot` to the issue or manually dispatch **Agents Auto-Pilot**.
Auto-Pilot invokes Agents 71 and the Agents 72 dispatch wrapper; those
components are not a replacement for ordinary intake.

Automation-created and reused pull requests must be ready for review. Do not
use draft state or PR closure to represent dependencies, staging, stack order,
or capacity. Use labels, PR-body lifecycle metadata, disabled auto-merge,
required checks, and exact-head guards.

## Source-Managed Changes

For shared behavior:

1. Fix the canonical workflow or template in `stranske/Workflows`.
2. Validate the Workflows template and sync contract there.
3. Deliver the matching consumer change through the managed sync path. A local
   forward fix is acceptable only when it matches the already-prepared source
   repair and closes an immediate safety gap.

Local product CI remains owned by this repository. Validate workflow edits with
Actionlint and run the focused workflow-contract tests before pushing.

See [`docs/AGENTS_POLICY.md`](../../docs/AGENTS_POLICY.md) and
[`docs/ops/codex-bootstrap-facts.md`](../../docs/ops/codex-bootstrap-facts.md)
for the protected-path and operator contracts.
