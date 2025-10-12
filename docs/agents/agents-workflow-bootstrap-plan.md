# Agents Workflow Bootstrap Plan

## Scope & Key Constraints
- `.github/workflows/agents-70-orchestrator.yml` is the **only** scheduled or
  manual automation surface.
- All automation routes through `reuse-agents.yml` /
  `reusable-70-agents.yml` so feature parity is preserved (readiness,
  watchdog, diagnostics, keepalive, bootstrap, issue verification).
- Concurrency at the workflow root must guard against overlapping runs on the
  same ref. Timeout enforcement continues to live inside the reusable stack,
  which caps downstream execution at 30 minutes.

## Acceptance Criteria / Definition of Done
1. Orchestrator workflow continues to expose the manual inputs listed in
   `docs/ci/WORKFLOWS.md` and fans into `reusable-70-agents.yml` without
   introducing bespoke JSON parsing layers.
2. Orchestrator declares per-ref concurrency guards with
   `cancel-in-progress: true` and delegates timeout coverage to the reusable
   workflow.
3. Documentation (CONTRIBUTING, `docs/ci/WORKFLOWS.md`, and these notes)
   describes the orchestrator as the sole entry point and links to monitoring
   guidance (48-hour quiet window, `ci-failure` tagging).
5. Tests in `tests/test_workflow_agents_consolidation.py` and
   `tests/test_workflow_naming.py` enforce the manual-only status, concurrency
   guards, and naming policy.

## Initial Task Checklist
- [x] Inventory orchestration features to confirm the orchestrator covers
  readiness, watchdog, diagnostics, bootstrap, verification, and keepalive
  paths without the consumer.
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
