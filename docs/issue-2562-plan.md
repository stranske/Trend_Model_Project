# Issue #2562 – Retire Legacy `agent-watchdog.yml`

## Scope and Key Constraints
- Remove the standalone `.github/workflows/agent-watchdog.yml` workflow or disable all of its triggers while leaving a stub that documents the deprecation.
- Ensure the orchestrator workflow (`.github/workflows/agents-70-orchestrator.yml`) remains the sole automation path for watchdog checks via the `params_json.enable_watchdog` flag that feeds `reusable-16-agents.yml`.
- Update archival documentation (`ARCHIVE_WORKFLOWS.md`) to record the retirement decision, rationale, and verification date.
- Avoid regressing other orchestrator paths (assign, consumer, readiness) and preserve existing workflow permissions, secrets, and dispatch inputs.
- Coordinate with Gate/GitHub Actions owners before permanently deleting shared components or secrets.

## Acceptance Criteria / Definition of Done
1. `agent-watchdog.yml` is either removed from `.github/workflows/` or left as a non-triggering stub that clearly states the workflow is retired.
2. Orchestrator watchdog executions continue to work when `enable_watchdog` resolves to `true`, confirmed via a dry-run or recent successful run log.
3. `ARCHIVE_WORKFLOWS.md` documents the retirement with the issue number, date, and validation notes referencing the orchestrator replacement path.
4. Repository CI (Gate and related required checks) remains green after the change set.
5. Any secrets or environment variables unique to the legacy workflow are either repurposed by the orchestrator or explicitly confirmed unused.

## Initial Task Checklist
- [ ] Inventory the current state of `.github/workflows/agent-watchdog.yml` and confirm no other workflows depend on it.
- [ ] Review orchestrator configuration to verify the `enable_watchdog` branch invokes the reusable watchdog job without relying on the legacy workflow.
- [ ] Implement the workflow removal or trigger disablement, ensuring a clear deprecation header remains if the file stays in the repo.
- [ ] Run or inspect an orchestrator workflow execution with `enable_watchdog=true` to validate watchdog coverage.
- [ ] Update `ARCHIVE_WORKFLOWS.md` with retirement details (issue, date, verification steps).
- [ ] Audit repository secrets/vars referenced by the legacy workflow and document any follow-up actions if they can now be deleted.
- [ ] Submit PR, monitor Gate (and other required) checks, and confirm no regressions.
