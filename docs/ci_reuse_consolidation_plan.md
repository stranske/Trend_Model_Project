# CI Workflow Consolidation Plan — Post Issue #2190 Snapshot

Last updated: 2026-10-12

Issue #2190 completed the consolidation roadmap that began in #1166/#1259. The repository now exposes only the reusable
workflows required by the trimmed automation surface.

## Current State
- Only four reusable workflows remain (`reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-92-autofix.yml`,
  `reusable-70-agents.yml`). The self-test matrix now lives in the manual-only
  `selftest-80-reusable-ci.yml` workflow.
- Visible workflows in the Actions tab were reduced to the final set documented in `WORKFLOW_AUDIT_TEMP.md` and `docs/ci/WORKFLOWS.md`.
- All auxiliary wrappers (gate orchestrators, labelers, watchdog forwards, etc.) were deleted, with `agents-43-codex-issue-bridge.yml` later reinstated to restore label-driven Codex automation.

## Completed Consolidation Actions
| Area | Action |
|------|--------|
| Agent automation | Removed `agents-41*`, `agents-42-watchdog.yml`, `agents-44-copilot-readiness.yml`, and `agents-45-verify-codex-bootstrap-matrix.yml`; consolidated around `agents-70-orchestrator.yml` + `reusable-70-agents.yml`, then reinstated `agents-43-codex-issue-bridge.yml` to preserve label-driven Codex bootstraps.
| Maintenance | Deleted legacy hygiene/self-test workflows (`maint-31`, `maint-34`, `maint-37`, `maint-38`, `maint-43`, `maint-44`, `maint-45`, `maint-48`, `maint-49`, `maint-52`, `maint-60`) and introduced the self-test wrapper that ultimately became the archived `Old/workflows/selftest-81-maint-wrapper.yml`. `agents-63-chatgpt-issue-sync.yml` was later reinstated (2025-10-07) to preserve issue fan-out from curated topic lists and is now guarded by tests. |
| PR checks | Removed auxiliary PR workflows (gate orchestrator, labeler, workflow lint, CodeQL, dependency review, path labeler) to align with the two final checks.

## Follow-Up Guardrails
- `tests/test_workflow_naming.py` enforces the `<area>-<NN>-<slug>.yml` convention and inventory coverage.
- `tests/test_workflow_agents_consolidation.py` verifies the orchestrator inputs and ensures legacy agent workflows do not return.
- `docs/ci/WORKFLOWS.md` is the authoritative description of the remaining automation footprint.

## Future Considerations
1. Keep the `selftest-8X-*` workflows manual—self-tests should only run when dispatched from the Actions UI.
2. Revisit CodeQL or dependency review if security tooling is reintroduced in a dedicated follow-up issue.
3. Validate external consumers when adjusting inputs on `reusable-10-ci-python.yml` or `reusable-12-ci-docker.yml`.

No additional consolidation actions are planned at this time.
