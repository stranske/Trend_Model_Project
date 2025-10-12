# Agents Consumer Workflow (Retired)

## Scope and Key Constraints
- Keep `.github/workflows/agents-62-consumer.yml` available for **manual**
  dispatch only. All automated triggers stay removed (no cron, push, or
  issue-driven runs).
- Preserve feature parity with `reuse-agents.yml` so the consumer continues to
  proxy readiness, watchdog, diagnostics, bootstrap, keepalive, and verification
  toggles via the `params_json` payload.
- Maintain the workflow-level concurrency guard scoped to
  `agents-62-consumer-${{ github.ref }}` with `cancel-in-progress: true` to prevent
  back-to-back manual dispatch collisions.
- Timeout enforcement for the reusable agents fan-out lives inside
  `reuse-agents.yml` → `reusable-70-agents.yml`; the consumer should not attempt
  to set `timeout-minutes` on the reusable-call job.
- Documentation must highlight that the orchestrator is the only scheduled
  automation surface and that post-change monitoring requires a 48-hour quiet
  window (tagging the source issue with `ci-failure`).

## Acceptance Criteria / Definition of Done
- `agents-62-consumer.yml` exposes only the `workflow_dispatch` trigger and keeps
  the concurrency guard at the workflow root.
- Manual runs continue to default to readiness + watchdog while treating
  bootstrap, preflight, keepalive, and verification features as explicit opt-ins
  via `params_json`.
- The dispatch job calls `reuse-agents.yml` without declaring an unsupported
  `timeout-minutes` value; explanatory comments document that timeouts are
  enforced downstream.
- `docs/ci/WORKFLOWS.md`, `docs/agents/agents-workflow-bootstrap-plan.md`, and
  these notes reflect the manual-only status, describe the orchestrator as the
  scheduled automation entry point, and outline the 48-hour monitoring
  expectation.
- Guard tests `tests/test_workflow_agents_consolidation.py` and
  `tests/test_workflow_naming.py` are re-run to validate structure and naming.

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
