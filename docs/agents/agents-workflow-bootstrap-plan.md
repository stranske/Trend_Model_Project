# Agents Workflow Bootstrap Plan

## Scope & Key Constraints
- `agents-consumer.yml` has been retired; all Codex automation now funnels
  through `.github/workflows/agents-70-orchestrator.yml`.
- The orchestrator must remain the single dispatch point for readiness,
  watchdog, diagnostics, and bootstrap flows while delegating execution to
  `reuse-agents.yml`.
- Manual and scheduled runs share the same configuration surface (standard
  inputs + `options_json`), so defaults must stay safe for unattended cron
  executions.
- Concurrency at the orchestrator level must guard against overlapping runs on
  the same ref, and each reusable job should retain explicit timeouts to avoid
  hung automation.

## Acceptance Criteria / Definition of Done
1. Orchestrator workflow exposes a concise `workflow_dispatch` interface that
   maps directly to `reuse-agents.yml` inputs without requiring auxiliary JSON
   parsing layers.
2. Workflow declares `concurrency: { group: agents-orchestrator-${{ github.ref }},
   cancel-in-progress: true }` to serialize activity per ref.
3. Delegated job (`Dispatch Agents Toolkit`) enforces a 30 minute timeout and
   surfaces downstream readiness, preflight, keepalive, and bootstrap statuses
   in the Actions UI.
4. Documentation (CONTRIBUTING, `docs/ci/WORKFLOWS.md`) references the
   orchestrator as the sole automation entry point and links to the reusable
   workflow for implementation details.
5. `reuse-agents.yml` continues to emit the `issue_numbers_json` and
   `first_issue` outputs for bootstrap consumers without schema changes.

## Initial Task Checklist
- [x] Remove `agents-consumer.yml` and update dependent documentation to point
  at the orchestrator.
- [x] Add concurrency and job timeouts to
  `.github/workflows/agents-70-orchestrator.yml`.
- [x] Confirm orchestrator inputs cover readiness, preflight, verification,
  bootstrap, diagnostics, and keepalive toggles.
- [x] Re-run workflow guard tests (`pytest tests/test_workflow_agents_consolidation.py
  tests/test_workflow_naming.py`) to ensure the naming and structure stay in
  sync.

## Verification Log

- 2024-06-15 – Consolidation updates verified with the agent workflow guard
  tests listed above; documentation now highlights the orchestrator as the
  single automation entry point.
