# Maint Post CI & Failure Tracker Hygiene Plan

## Scope and Key Constraints
- **Maintain existing automation**: Keep the Maint Post CI aggregator and Failure Tracker workflows intact while updating logic for renamed workflows; avoid regressions to other automation relying on these outputs.
- **Deduplicated communications**: Ensure both the PR summary comment and the failure-tracker issues remain single-source-of-truth, preventing duplicate posts when workflow names change or jobs rerun.
- **Compatibility with workflow renames**: Treat job and workflow identifiers as dynamic; avoid hard-coded names so future renames do not require manual updates.
- **Auditability**: Preserve traceable links (workflow run URLs, issue references) within generated content for post-mortem reviews.
- **Testability**: Prefer deterministic unit or integration tests and, where not feasible, document manual validation procedures for failure scenarios.
- **Non-disruptive rollout**: Stage changes behind feature-flag-like configuration where possible, and validate in a sandbox PR before shipping to production branches.

## Acceptance Criteria / Definition of Done
1. Maint Post CI leaves a single summary comment per PR even across reruns, referencing the correct job names and URLs after workflow renames.
2. Failure Tracker opens or updates exactly one GitHub issue per failure category despite workflow/job renames; historical duplicates are cleaned or linked.
3. Updated summaries and failure issues reflect the new workflow naming conventions across tables, headings, and deep links.
4. Automated or manual tests cover the name-derivation logic and deduplication safeguards; documentation describes how to run them.
5. Evidence links to one successful Maint Post CI run and one Failure Tracker update are captured for audit once changes deploy.
6. Rollout notes outline back-out steps in case regressions are discovered post-merge.

## Initial Task Checklist
- [ ] Inventory current Maint Post CI and Failure Tracker workflow files, identifying all hard-coded job or workflow names.
- [ ] Map the new workflow names to their corresponding jobs; define a normalization strategy for identifiers used in summaries and failure categories.
- [ ] Update Maint Post CI summary generation logic to consume dynamic job metadata and enforce single-comment deduplication.
- [ ] Adjust Failure Tracker categorization keys to rely on normalized workflow identifiers; ensure updates target existing issues when names change.
- [ ] Extend or add automated tests covering renamed-workflow scenarios for both systems.
- [ ] Prepare manual validation steps, including creating an intentional failure PR to confirm tracker behavior.
- [ ] Document evidence collection process and rollback steps in project notes.
