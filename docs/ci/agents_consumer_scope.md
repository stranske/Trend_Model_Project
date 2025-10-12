# Agents Consumer Workflow (Retired)

The legacy `.github/workflows/agents-consumer.yml` wrapper has been removed as
part of the Issue #2466 consolidation. Contributors should route every agent
automation task through [`agents-70-orchestrator.yml`](../../.github/workflows/agents-70-orchestrator.yml),
which fans into the reusable agents toolkit for readiness probes, diagnostics,
Codex bootstrap, keepalive, and watchdog sweeps.

## Replacement Flow

1. Navigate to **Actions → Agents 70 Orchestrator → Run workflow**.
2. Provide the desired inputs (branch, readiness toggles, bootstrap settings,
   and any overrides in the `options_json` payload).
3. Review the `orchestrate` job summary for readiness tables, bootstrap status,
   and keepalive signals.

The JSON examples that previously lived in this file can now be found in the
orchestrator documentation:

- [`docs/ci/WORKFLOWS.md`](WORKFLOWS.md) – canonical workflow roster and manual
  dispatch payloads.
- [`docs/WORKFLOW_GUIDE.md`](../WORKFLOW_GUIDE.md) – topology guide describing
  the orchestrator-only automation model.
- [`docs/agent-automation.md`](../agent-automation.md) – deep dive into
  orchestrator inputs, troubleshooting, and telemetry.

If you discover a reference to `agents-consumer.yml` elsewhere in the
repository, update it to point to the orchestrator so the documentation remains
consistent with the simplified topology.
