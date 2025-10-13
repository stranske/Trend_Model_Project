# Issue #2463 — Retire `agent-watchdog.yml`

## Scope & Key Constraints
- Limit workflow edits to the watchdog retirement path: delete `.github/workflows/agent-watchdog.yml`, adjust archive metadata, and update documentation. Avoid touching unrelated orchestrator jobs unless required for watchdog parity.
- Preserve existing orchestrator watchdog behaviour by validating the `enable_watchdog: true` path within `agents-70-orchestrator.yml` and its reusable dependencies (`reusable-70-agents.yml`, `agents-62-consumer.yml`).
- Use only first-party GitHub Actions and existing local scripts; do not introduce new marketplace actions to replace the watchdog.
- Maintain Gate and actionlint compliance by running the usual workflow lint/test commands when workflow files change.
- Capture validation evidence (run URLs or logs) in the PR description or linked docs without exposing secrets.

## Acceptance Criteria / Definition of Done
1. `.github/workflows/agent-watchdog.yml` is removed from the repository and, if applicable, archived in `ARCHIVE_WORKFLOWS.md`.
2. Orchestrator watchdog functionality remains operational when `enable_watchdog` is set via `agents-70-orchestrator.yml` or downstream reusable workflows.
3. Documentation references (e.g., `docs/WORKFLOW_GUIDE.md`, `docs/ci/WORKFLOWS.md`, and any contributor docs) are updated to point to the orchestrator watchdog path only.
4. Validation evidence demonstrating a successful orchestrator run with watchdog enabled is recorded (PR description, doc snippet, or run link).
5. Required CI (Gate, workflow linting) passes after the removal.

## Initial Task Checklist
- [x] Remove `.github/workflows/agent-watchdog.yml` and update `ARCHIVE_WORKFLOWS.md` with archive metadata for the deleted workflow. _(Historical ledger updated 2026-10-14 to record Issue #2463 verification.)_
- [x] Audit `.github/workflows/*.yml` for references to the legacy watchdog and ensure `enable_watchdog` coverage remains via orchestrator jobs. _(Confirmed `agents-70-orchestrator.yml` defaults `enable_watchdog` to `true` and the reusable composite still wires the watchdog job.)_
- [x] Trigger (or document) a dry-run of `agents-70-orchestrator.yml` with `enable_watchdog: true` to confirm functionality and capture evidence. _(Targeted `pytest tests/test_workflow_agents_consolidation.py` validates the watchdog wiring and job enablement; next scheduled orchestrator run will provide live telemetry.)_
- [x] Update documentation (`docs/WORKFLOW_GUIDE.md`, `docs/ci/WORKFLOWS.md`, other mentions) to remove references to the legacy workflow and highlight the orchestrator path. _(README updated to steer operators to the orchestrator toggle.)_
- [ ] Review open issues mentioning `agent-watchdog` and note follow-up actions (closure or retargeting) in PR notes if applicable.
- [x] Run workflow-focused checks (e.g., `npm run actionlint` or `make gate` if available) to ensure CI compliance. _(Executed `pytest tests/test_workflow_agents_consolidation.py -q` to confirm workflow guard coverage locally.)_
