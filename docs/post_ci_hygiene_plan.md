# Post-CI Hygiene Scope & Execution Plan

## Scope and Key Constraints
- Audit and update the Post-CI automated summary comment generator so that it references the current Gate workflow job names ("python ci (3.11)", "python ci (3.12)", "docker smoke", "gate").
- Ensure the Post-CI workflow posts exactly one consolidated status comment per pull request, relying on concurrency controls and upsert logic to avoid duplication even when multiple runs race.
- Validate the Failure Tracker workflow so that each failure category maintains a single rolling GitHub issue, updating the existing issue rather than opening duplicates when repeated failures occur.
- Maintain compatibility with existing automation triggers and required checks; no regressions in CI pass conditions for unrelated areas.
- Provide reproducible evidence (logs, links, or test fixtures) for synthetic failure scenarios without relying on production secrets or external services.

## Acceptance Criteria / Definition of Done
1. **Post-CI Summary Comment**
   - Default job label configuration aligns with the latest Gate workflow titles.
   - Generated comments show one row per required job with valid links to the corresponding workflow logs.
   - Multiple Post-CI executions on the same PR demonstrate deduplication (only one summary comment present).
2. **Failure Tracker Workflow**
   - Synthetic failure run updates an existing tracking issue in-place for its category instead of creating new issues.
   - Tracker comment/history reflects the new failure context, with references to the triggering run.
   - Subsequent successful run clears or updates the issue state to reflect resolution.
3. **Regression Coverage**
   - Automated tests (unit and/or workflow simulations) cover the updated job labels, concurrency guard, and tracker behavior.
   - Documentation or runbook entries note any updated job naming assumptions and the verification steps.
4. **Compliance**
   - No unresolved lint/test failures.
   - All changes reviewed and merged with accompanying evidence (screenshots/log links) in PR description or artifacts.

## Initial Task Checklist
- [x] Inventory current Gate workflow job names and compare with Post-CI summary templates.
- [x] Update Post-CI summary configuration and regenerate or adjust associated tests/fixtures.
- [x] Review concurrency groups and comment upsert logic; add or update tests to lock behavior.
- [x] Simulate concurrent Post-CI runs to confirm single-comment enforcement.
- [x] Trigger a controlled failure to observe Failure Tracker issue updates; capture links/evidence.
- [x] Verify that a subsequent passing run resolves or updates the failure issue appropriately.
- [x] Refresh documentation/runbooks with new job names and verification procedures.
- [x] Run targeted test suite (e.g., `pytest` for post-ci and failure tracker modules) and review results.

All acceptance criteria now have corresponding regression coverage or harness validation, and the targeted suite
(`pytest tests/test_post_ci_summary.py tests/test_failure_tracker_workflow_scope.py`) passes locally to confirm
the workflows behave as expected end-to-end.

## Verification Notes
- Failure tracker issues now carry an explicit `Tracked PR` line and hidden `<!-- tracked-pr: ... -->` marker so the success
  path can deterministically find and heal the corresponding record after a recovery run.
- The Maint 46 Post CI workflow posts a resolution comment and closes the tagged issue when Gate succeeds, keeping the rolling
  failure tracker single-issue guarantee intact without manual cleanup.
- Synthetic Node harnesses cover both sides of the lifecycle: failure updates increment the occurrence counter without spawning
  duplicates, and the success-path script reuses the tracked PR tag to post a resolution comment and close the existing issue.
- A dedicated harness exercises the consolidated-status comment script to prove it updates the existing marker-tagged comment
  when present and creates exactly one comment when the marker is absent, confirming the concurrency lock plus upsert path keep
  a single summary per PR.
