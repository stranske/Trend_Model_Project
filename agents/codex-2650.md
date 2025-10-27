# Issue #2650 — Transition to Orchestrator + Issue Intake

## Checklist for Codex Orchestrator/Bridge Workflows

### Orchestrator/Intake Workflows
- [x] Verified `.github/workflows/agents-70-orchestrator.yml` remains the single execution surface and documentation now calls out the orchestrator-only topology.
- [x] Confirmed `.github/workflows/agents-63-issue-intake.yml` stays wired to the orchestrator hand-off with no fallback consumer hooks.

### Trigger Labels
- [x] Updated guidance so the Agent task issue template explicitly documents the auto-applied `agents` and `agent:codex` labels.
- [x] Added regression coverage to ensure the intake front continues to listen for `agent:codex` and that the template keeps both labels.

### Acceptance Criteria
- [x] Creating an issue with the Agent task template applies the correct labels, triggering the Codex intake workflow to open a branch/PR that flows into the orchestrator run.
- [x] No legacy consumer workflows remain on disk (confirmed by guard tests and the archive ledger).
- [x] Documentation refreshed across the workflow README, workflow guide, and Codex bootstrap facts to reflect the orchestrator-only path codified by Issue #2650.

## Notes
- Agent template text now points authors at the orchestrator entry point and clarifies that the intake workflow handles branch + PR setup once the labels land.
- `ARCHIVE_WORKFLOWS.md` records the Issue #2650 verification so future audits can trace when the consumer shim was last validated as removed.
- Tests in `tests/test_workflow_agents_consolidation.py` cover both the template labels and the intake trigger conditions to guard against regressions.
