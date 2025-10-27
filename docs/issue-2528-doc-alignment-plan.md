# Documentation Alignment Plan for Issue #2528

## Scope and Key Constraints
- Update project documentation to reflect the finalized Orchestrator-centric automation topology across all relevant guides.
- Ensure references to agent entry points, triggers, and workflow names are current for the Orchestrator, Issue Bridge, and Verify Assignment flows.
- Capture archival information for retired workflows (e.g., legacy watchdog, self-tests) while maintaining traceability to their replacements.
- Refresh contributor guidance so that onboarding steps reference the Gate check, workflow dispatch usage, and health self-check outputs.
- Minimize disruption to existing documentation structure; changes should be incremental edits rather than full rewrites.
- Adhere to repository documentation style conventions (markdown headings, tables, existing section hierarchy).

## Acceptance Criteria / Definition of Done
1. **Agents.md** accurately describes the current agent ecosystem:
   - Includes entries for Orchestrator, Issue Intake, and Verify Agent Assignment with up-to-date names.
   - Lists trigger types (schedule, workflow_dispatch, issue_comment, etc.) and manual dispatch instructions where applicable.
   - Provides cross-links or references to relevant workflows or configs if they exist in the repo.
2. **ARCHIVE_WORKFLOWS.md** contains an "Archived" section that:
   - Documents legacy watchdog and other retired self-test workflows.
   - Notes replacement workflows or rationale for deprecation.
   - Maintains consistent formatting with existing workflow ledger entries.
3. **CONTRIBUTING.md** guidance reflects current automation practices:
   - Mentions Gate as the mandatory PR check and how contributors can verify it locally or via CI.
   - Explains how to trigger maintenance and agent workflows via `workflow_dispatch` and when to do so.
   - Directs readers to the health self-check output within GitHub Actions run summaries.
4. All modified documents pass markdown linting (if applicable) and render without structural regressions.
5. Stakeholders (e.g., docs or automation maintainers) sign off on the updates via PR review.

## Initial Task Checklist
- [ ] Audit existing sections in Agents.md for outdated workflow names, triggers, or missing entries related to Orchestrator, Issue Bridge, and Verify Assignment.
- [ ] Gather authoritative workflow metadata (YAML triggers, dispatch parameters) from `.github/workflows/` to inform documentation updates.
- [ ] Draft revised sections in Agents.md, ensuring consistent formatting and inclusion of manual dispatch steps.
- [ ] Review ARCHIVE_WORKFLOWS.md for current structure; design and insert an "Archived" section capturing legacy watchdog and retired self-tests.
- [ ] Confirm replacement workflows or references for archived items and document them alongside deprecations.
- [ ] Update CONTRIBUTING.md to cover Gate requirements, workflow_dispatch usage, and locating health self-check outputs.
- [ ] Perform proofreading and run markdown/style checks if available.
- [ ] Circulate changes for review and incorporate feedback prior to merge.
