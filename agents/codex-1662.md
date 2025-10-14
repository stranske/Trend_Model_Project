# Issue #1662 – Standardize Issue→Agent Assignment and Watchdog Checks

> **Note (2026-10-12):** Issue #2190 replaced the `agents-41-*` family with `agents-70-orchestrator.yml`. The historical checklist below remains for context but is no longer the live architecture.

## Workflow consolidation
- [x] Audit existing agent workflows in `.github/workflows/` and record triggers/shared steps. (See `docs/ci/WORKFLOW_SYSTEM.md` inventory.)
- [x] Draft unified architecture for `agents-41-assign-and-watch.yml`, covering triggers, reusable calls, and permissions. (Superseded by `agents-70-orchestrator.yml`.)

## Implementation
- [x] Implement label-driven handler that routes assignments through `reusable-90-agents.yml` readiness checks before assigning. (Historical; now handled by `reusable-16-agents.yml`.)
- [x] Add logic to clear assignments when agent labels are removed.
- [x] Implement scheduled watchdog sweep that pings stale owners and escalates when unavailable.
- [x] Share readiness utilities via `reusable-90-agents.yml` for both assignment and sweep paths. (Historical; now handled by `reusable-16-agents.yml`.)

## Migration & documentation
- [x] Convert legacy `agents-41-assign.yml` and `agents-42-watchdog.yml` into thin wrappers around the unified workflow.
- [x] Update workflow documentation (`WORKFLOW_GUIDE.md`, `docs/agent-automation.md`, `docs/ci_reuse.md`, `docs/ops/codex-bootstrap-facts.md`) to describe consolidated behaviour.
- [x] Validate acceptance criteria via `tests/test_workflow_agents_consolidation.py` (ensures orchestrator exists, wrappers delegate, docs mention new flow).

## Notes
- The unified workflow exposes readiness outputs to downstream jobs and uploads watchdog summaries for observability.
- Guardrail test `tests/test_workflow_agents_consolidation.py` keeps consolidation invariants enforced in CI.
