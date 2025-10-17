# Codex Bootstrap: Consolidate PR Comment Self-Tests (Issue #2720)

[Source Issue #2720](https://github.com/stranske/Trend_Model_Project/issues/2720)

**Consolidation Decision:** Maint 46 Post CI remains the canonical automated PR comment surface; the Self-test Runner workflow
is available for manual annotations, and the legacy comment wrappers stay retired.

## TODO Checklist
- [x] Remove duplicate self-test workflows and record their status in `ARCHIVE_WORKFLOWS.md`.
- [x] Update CI documentation (`docs/ci/WORKFLOW_SYSTEM.md`, `docs/ci/WORKFLOWS.md`) to describe the consolidated path.
- [x] Add a regression guard (`tests/test_workflow_selftest_consolidation.py`) so archived wrappers cannot silently return.
- [x] Monitor Maint 46 Post CI runs after consolidation and backfill any missed edge cases. (Run [#18597199319](https://github.com/stranske/Trend_Model_Project/actions/runs/18597199319) validated the comment handoff.)

## Next Steps / Owners
- Workflow consolidation monitoring: @workflow-maintainer
- Documentation follow-through: @docs-lead
- Regression guard upkeep: @qa-owner

