# Gate Summary Job & Failure Tracker Hygiene Plan

## Scope and Key Constraints
- **Maintain existing automation**: Keep the Gate summary job aggregator and Failure Tracker workflows intact while updating logic for renamed workflows; avoid regressions to other automation relying on these outputs.
- **Deduplicated communications**: Ensure both the PR summary comment and the failure-tracker issues remain single-source-of-truth, preventing duplicate posts when workflow names change or jobs rerun.
- **Compatibility with workflow renames**: Treat job and workflow identifiers as dynamic; avoid hard-coded names so future renames do not require manual updates.
- **Auditability**: Preserve traceable links (workflow run URLs, issue references) within generated content for post-mortem reviews.
- **Testability**: Prefer deterministic unit or integration tests and, where not feasible, document manual validation procedures for failure scenarios.
- **Non-disruptive rollout**: Stage changes behind feature-flag-like configuration where possible, and validate in a sandbox PR before shipping to production branches.

## Acceptance Criteria / Definition of Done
1. The Gate summary job leaves a single summary comment per PR even across reruns, referencing the correct job names and URLs after workflow renames.
2. Failure Tracker opens or updates exactly one GitHub issue per failure category despite workflow/job renames; historical duplicates are cleaned or linked.
3. Updated summaries and failure issues reflect the new workflow naming conventions across tables, headings, and deep links.
4. Automated or manual tests cover the name-derivation logic and deduplication safeguards; documentation describes how to run them.
5. Evidence links to one successful Gate summary job run and one Failure Tracker update are captured for audit once changes deploy.
6. Rollout notes outline back-out steps in case regressions are discovered post-merge.

## Initial Task Checklist
- [x] Inventory current Gate summary job and Failure Tracker workflow files, identifying all hard-coded job or workflow names.
- [x] Map the new workflow names to their corresponding jobs; define a normalization strategy for identifiers used in summaries and failure categories.
- [x] Update Gate summary job summary generation logic to consume dynamic job metadata and enforce single-comment deduplication.
- [x] Adjust Failure Tracker categorization keys to rely on normalized workflow identifiers; ensure updates target existing issues when names change.
- [x] Extend or add automated tests covering renamed-workflow scenarios for both systems.
- [x] Prepare manual validation steps, including creating an intentional failure PR to confirm tracker behavior.
- [x] Document evidence collection process and rollback steps in project notes.

## Manual Validation Plan
- Exercise the Gate summary job on a sandbox PR after a workflow rename by rerunning Gate and confirming the consolidated comment updates the existing post instead of creating a duplicate.
- Trigger a controlled Gate job failure (e.g., force a lint failure) to validate the Failure Tracker reuses the existing issue signature after the rename and records the new occurrence count.
- Capture screenshots or markdown copies of the Gate summary job summary and failure tracker issue update for archival alongside the run URLs.

## Evidence Tracking
- Gate summary job run link: https://github.com/stranske/Trend_Model_Project/actions/runs/18480830490
- Failure Tracker issue update link: https://github.com/stranske/Trend_Model_Project/issues/2552

## Detailed Manual Validation Procedure

1. **Sandbox preparation**
   - Create a throwaway PR sourced from the feature branch with a trivial change (e.g., edit documentation) so the Gate summary job triggers without affecting production branches.
   - Confirm the new PR inherits the updated workflows by checking the workflow run summary.
2. **Gate summary job rename verification**
   - Rerun the Gate workflow to produce multiple workflow_run events.
   - Validate that the Gate summary job comment edits in place by comparing the comment ID before and after the rerun; no new comment should appear.
   - Confirm the required-check table lists the renamed job titles exactly as they appear in the Gate job metadata.
3. **Failure Tracker stability drill**
   - Introduce an intentional failure (for example, add a failing unit test or format violation guarded behind the sandbox PR) and rerun Gate until the workflow concludes with `failure`.
   - Inspect the failure-tracker issue search for the workflow slug and ensure the existing issue reopens/updates instead of spawning a duplicate.
   - Verify the issue body reflects the slugged workflow identifier (`workflowId`) and that the occurrence counter increments.
4. **Cleanup**
   - Remove the intentional failure and rerun Gate to ensure the tracker records the healing event and the ci-failure label drops from the PR.

## Evidence Collection & Rollback Notes

- **Evidence artifacts**: Capture the Gate summary job PR comment URL, the failure tracker issue permalink, and the corresponding GitHub Actions run URLs. Store the links in the section above once the sandbox validation completes.
- **Archival copies**: Download the Gate summary job comment body and failure tracker issue history as Markdown for offline audit. Attach them to the sprint retro notes or the incident tracker.
- **Rollback strategy**:
  - Revert the commits touching `tools/post_ci_summary.py` and the Gate summary job in `.github/workflows/pr-00-gate.yml` if regressions appear.
  - Restore the prior workflow artefacts by force-pushing the previous workflow commit and invalidating cached Docker layers if the issue impacts gating reliability.
  - Disable the failure tracker temporarily by setting `DISABLE_FAILURE_ISSUES: 'true'` in the workflow dispatch environment while investigating.
- **Monitoring checklist**: For the week following deployment, monitor Gate summary job summaries for duplicate comments and watch the failure tracker label volumes to spot unexpected issue bursts.
