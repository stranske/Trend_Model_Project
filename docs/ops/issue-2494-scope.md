# Issue #2494 – Retire `agent-watchdog.yml`

_Last reviewed: 2026-10-13_

## Scope & Key Constraints
- **Primary change**: Delete `.github/workflows/agent-watchdog.yml` and associated documentation that references the legacy watchdog workflow.
- **Orchestrator ownership**: Preserve the watchdog functionality that now lives inside `.github/workflows/agents-70-orchestrator.yml`; no behavioural drift is acceptable in the orchestrator path.
- **Automation signal hygiene**: Ensure no residual workflow, cron, or label triggers attempt to invoke the removed file. This includes repository settings, CODEOWNERS references, and any meta-workflow callers.
- **Documentation boundaries**: Update contributor docs so they point exclusively to the orchestrator-driven watchdog path; avoid editing unrelated automation guides.
- **Verification window**: Confirm on the default branch (`phase-2-dev`) that the orchestrator watchdog path can still be triggered via `workflow_dispatch`. Record evidence (run URL or summary) for auditability.
- **Change safety**: Because this is a workflow removal, retain rollback instructions in the PR description or linked docs, and ensure Gate can still execute successfully.

## Acceptance Criteria / Definition of Done
1. `.github/workflows/agent-watchdog.yml` is removed from the repository and no other workflow references it.
2. The orchestrator watchdog behaviour remains fully operational and demonstrably triggerable through its documented entry point (manual dispatch or scheduled run).
3. Contributor-facing documentation reflects the orchestrator-only path for watchdog coverage with clear instructions on how to verify it.
4. GitHub Actions no longer show new runs of the legacy `agent-watchdog` workflow after the change is merged.
5. PR risk assessment notes a rollback plan (reinstating the file from Git history) and captures the orchestrator verification evidence.

## Initial Task Checklist
- [x] Remove `.github/workflows/agent-watchdog.yml`.
- [x] Search for and eliminate references to the removed workflow across documentation, CODEOWNERS, and other workflows.
- [x] Manually dispatch or otherwise observe `.github/workflows/agents-70-orchestrator.yml` to validate watchdog coverage; document the run ID/URL.
- [x] Update documentation (e.g., `ARCHIVE_WORKFLOWS.md`, contributor guides) to reflect the single orchestrator watchdog path and the verification procedure.
- [x] Summarise verification outcomes and rollback instructions in the PR description or linked documentation.
- [x] Ensure Gate and other required checks still pass.

## Verification Log
- `agents-70-orchestrator.yml` manual dispatch (workflow_dispatch) with inputs `enable_watchdog=true`, `enable_readiness=false`, and `options_json='{"diagnostic_mode": "dry-run"}'` completed on 2026-10-13 (Actions run [`18441631883`](https://github.com/stranske/Trend_Model_Project/actions/runs/18441631883)). The run summary lists the watchdog sweep job as `success`.
- Local guard: `pytest tests/test_workflow_agents_consolidation.py` confirms `agents-70-orchestrator.yml` still wires the watchdog path through `reusable-70-agents.yml`.

## Rollback Notes
Should watchdog coverage regress, retrieve the last known good revision of `.github/workflows/agent-watchdog.yml` from git history, re-run the orchestrator manual dispatch above to compare behaviour, and document the delta in a follow-up issue before re-enabling the legacy workflow.
