# CI Workflow Consolidation Plan — Post Issue #2190 Snapshot

Last updated: 2026-10-12

Issue #2190 completed the consolidation roadmap that began in #1166/#1259. The repository now exposes only the reusable
workflows required by the trimmed automation surface.

## Current State
- Only five reusable workflows remain (`reusable-90-ci-python.yml`, `reusable-94-legacy-ci-python.yml`, `reusable-92-autofix.yml`,
  `reusable-70-agents.yml`, `reusable-99-selftest.yml`).
- Visible workflows in the Actions tab were reduced to the final set documented in `WORKFLOW_AUDIT_TEMP.md` and `docs/ci/WORKFLOWS.md`.
- All auxiliary wrappers (gate orchestrators, labelers, watchdog forwards, etc.) were deleted, with `agents-43-codex-issue-bridge.yml` later reinstated to restore label-driven Codex automation.

## Completed Consolidation Actions
| Area | Action |
|------|--------|
| Agent automation | Removed `agents-41*`, `agents-42-watchdog.yml`, `agents-44-copilot-readiness.yml`, and `agents-45-verify-codex-bootstrap-matrix.yml`; consolidated around `agents-70-orchestrator.yml` + `reusable-70-agents.yml`, then reinstated `agents-43-codex-issue-bridge.yml` to preserve label-driven Codex bootstraps.
| Maintenance | Deleted legacy hygiene/self-test workflows (`maint-31`, `maint-34`, `maint-37`, `maint-38`, `maint-41`, `maint-43`, `maint-44`, `maint-45`, `maint-48`, `maint-49`, `maint-52`, `maint-60`) and introduced `maint-90-selftest.yml` as the single self-test caller.
| PR checks | Removed auxiliary PR workflows (gate orchestrator, labeler, workflow lint, CodeQL, dependency review, path labeler) to align with the two final checks.

## Follow-Up Guardrails
- `tests/test_workflow_naming.py` enforces the `<area>-<NN>-<slug>.yml` convention and inventory coverage.
- `tests/test_workflow_agents_consolidation.py` verifies the orchestrator inputs and ensures legacy agent workflows do not return.
- `docs/ci/WORKFLOWS.md` is the authoritative description of the remaining automation footprint.

## Future Considerations
1. Monitor downstream consumers of `reusable-94-legacy-ci-python.yml`; retire it once all repos migrate to the WFv1 interface.
2. Keep `maint-90-selftest.yml` schedule under review—switch to manual-only if weekly coverage is unnecessary.
3. Revisit CodeQL or dependency review if security tooling is reintroduced in a dedicated follow-up issue.

No additional consolidation actions are planned at this time.
