# Agents Consumer Workflow – Planning Notes (Issue #2464 Refresh)

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

## Behaviour inventory (2026-10-12 audit)

| Capability | Consumer implementation | Orchestrator parity | Notes |
| --- | --- | --- | --- |
| Readiness probe | Defaults to enabled via `params_json` | Enabled via `enable_readiness` input | Both paths call `reuse-agents.yml` → `reusable-70-agents.yml`; no divergence.
| Watchdog sweep | Defaults to enabled | Enabled through `enable_watchdog` input | Reusable workflow enforces identical keepalive/watchdog steps and timeout coverage.
| Codex preflight | Opt-in flag in JSON payload | `enable_preflight` manual input | Toggle forwards unchanged; reusable workflow owns diagnostics.
| Bootstrap PR fan-out | Opt-in (`enable_bootstrap`) with optional label override | `options_json.enable_bootstrap` + label input | Shared bootstrap job fans out PRs and applies labels identically.
| Issue verification | Opt-in (`enable_verify_issue`/`verify_issue_number`) | Manual input pair | Uses same verification job inside reusable stack.
| Keepalive dispatch | Opt-in JSON toggle | `options_json.enable_keepalive` | Converges on shared keepalive job; defaults match.
| Diagnostics / dry run | `options_json.diagnostic_mode` (manual payload) | Same JSON payload forwarded | Diagnostic knobs centralised; consumer adds no unique handling.

**Conclusion:** every capability surfaced by the consumer exists in the
orchestrator + reusable toolkit stack. The consumer remains solely as a manual
JSON entry point; removing it would not drop functionality beyond that input
surface.

## Initial Task Checklist
1. Inventory the consumer vs orchestrator inputs to confirm no unique
   functionality is lost by consolidating into the reusable toolkit.
2. Ensure the workflow `on:` section remains limited to `workflow_dispatch` and
   that concurrency is scoped by ref with `cancel-in-progress: true`.
3. Verify the dispatch job delegates to `reuse-agents.yml` with the same output
   mapping and without `timeout-minutes` overrides.
4. Update documentation to explain the manual-only status, how to use
   `params_json`, and the post-change monitoring plan.
5. Execute `pytest tests/test_workflow_agents_consolidation.py
   tests/test_workflow_naming.py` to confirm guardrails stay green before
   shipping changes.
