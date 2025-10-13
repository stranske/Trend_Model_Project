# Agents Workflow Bootstrap Plan

## Scope & Key Constraints
- `.github/workflows/agents-70-orchestrator.yml` is the **only** scheduled
  automation surface. Manual dispatch remains available in both the
  orchestrator and the legacy `agents-62-consumer.yml` wrapper.
- `agents-62-consumer.yml` exists solely for curated manual runs that need the
  JSON `params_json` input surface. All automated triggers (cron, issue
  events) stay disabled.
- Both dispatchers must converge on `reusable-70-agents.yml` so feature parity is preserved (readiness,
  watchdog, diagnostics, keepalive, bootstrap, issue verification).
- Concurrency at the workflow root must guard against overlapping runs on the
  same ref. Timeout enforcement continues to live inside the reusable stack,
  which caps downstream execution at 30 minutes.

## Acceptance Criteria / Definition of Done
1. Orchestrator workflow continues to expose the manual inputs listed in
   `docs/ci/WORKFLOWS.md` and fans into `reusable-70-agents.yml` without
   introducing bespoke JSON parsing layers.
2. Both orchestrator and consumer workflows declare per-ref concurrency guards
   with `cancel-in-progress: true` and delegate timeout coverage to the
   reusable workflow.
3. `agents-62-consumer.yml` remains manual-only and retains the
   `params_json`-driven defaults for readiness + watchdog, with bootstrap,
   preflight, verification, and keepalive staying opt-in.
4. Documentation (CONTRIBUTING, `docs/ci/WORKFLOWS.md`, and these notes)
   describes the orchestrator as the scheduled entry point, calls out the
   manual-only consumer surface, and links to monitoring guidance (48-hour
   quiet window, `ci-failure` tagging).
5. Tests in `tests/test_workflow_agents_consolidation.py` and
   `tests/test_workflow_naming.py` enforce the manual-only status, concurrency
   guards, and naming policy.

## Initial Task Checklist
- [x] Inventory orchestration features to confirm the orchestrator covers
  readiness, watchdog, diagnostics, bootstrap, verification, and keepalive
  paths without the consumer.
- [x] Keep `agents-62-consumer.yml` manual-only with
  `concurrency: agents-62-consumer-${{ github.ref }}` and surface parity with the
  reusable toolkit via `reusable-70-agents.yml`.
- [x] Document manual dispatch expectations and the post-change monitoring
  window in `docs/ci/WORKFLOWS.md`.
- [x] Re-run workflow guard tests (`pytest tests/test_workflow_agents_consolidation.py
  tests/test_workflow_naming.py`) to ensure the naming and structure stay in
  sync.

## Verification Log

- 2024-06-15 – Consolidation updates verified with the agent workflow guard
  tests listed above; documentation now highlights the orchestrator as the
  primary scheduled automation entry point.
- 2026-10-12 – Issue #2464 audit: consumer cron/issue triggers removed,
  concurrency guards validated, docs updated with monitoring guidance, and
  workflow guard tests re-run.
