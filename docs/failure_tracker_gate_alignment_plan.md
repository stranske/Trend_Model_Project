# Failure Tracker Gate Integration Plan (Issue #2436)

## Scope & Key Constraints
- Repoint the failure-tracker automation to listen for `workflow_run` events emitted by the **Gate** workflow only; do not widen event scope beyond Gate or alter other workflow triggers.
- Consolidate tracker update logic into the existing `maint-post-ci.yml` pipeline while keeping backward-compatible hooks for the legacy tracker workflow until removal is validated.
- Maintain single-source-of-truth updates for the failure tracker issue and PR comment; avoid introducing additional bot comment pathways.
- Preserve existing rate limits, cooldown handling, and deduplication semantics already defined in the tracker to prevent regression in issue/comment noise.
- Ensure CI label management remains idempotent and aligns with repository labeling conventions (e.g., create `ci-failure` on demand without duplicating other labels).
- Operate within GitHub Actions constraints (no new third-party actions without security review; reuse in-repo composite actions or patterns where possible).
- Provide observability (logs/artifacts) sufficient to validate Gate integration without enabling noisy verbose modes by default.

## Acceptance Criteria / Definition of Done
1. Failure tracker no longer reacts to runs from PR 10 or PR 12 (historical duplicates) once Gate rerouting is active.
2. On any qualifying Gate failure:
   - `maint-post-ci.yml` orchestrates the tracker update sequence and produces exactly one updated issue entry and one persistent PR comment.
   - No additional bot comments appear on the PR from other workflows.
3. Successful application (and removal on recovery) of the `ci-failure` label via the consolidated workflow.
4. Legacy failure-tracker workflow (if retained temporarily) delegates cleanly or is safely disabled without breaking detection.
5. Telemetry/artifacts confirm the workflow path taken (Gate → Maint Post-CI → tracker update) for auditability.
6. Documentation/config references updated to reflect the new event source and workflow responsibilities.

## Initial Task Checklist
- [ ] Audit current `.github/workflows/maint-33-check-failure-tracker.yml` (or successor) to catalog tracker update logic, shared environment variables, and comment/issue handling.
- [ ] Update the failure-tracker workflow trigger to `on: workflow_run` for `workflows: ["Gate"]` with `types: [completed]`; ensure proper filtering for branch/PR scope.
- [ ] Refactor tracker update steps into `maint-post-ci.yml`, preserving inputs, secrets, and retry/backoff behavior; leave a thin shell in the original workflow if needed for compatibility.
- [ ] Wire `maint-post-ci.yml` to invoke the tracker update job conditionally based on Gate outcomes (failed, cancelled, neutral).
- [ ] Implement `ci-failure` label management in `maint-post-ci.yml` (apply on failure, remove when passing) with existing helper scripts or GitHub CLI usage.
- [ ] Verify that duplicate comment suppression still functions when updates originate from the consolidated workflow (reuse existing markers/timestamps).
- [ ] Exercise dry-run or sandbox runs referencing PRs 10 and 12 to confirm no new events fire from those historical workflows.
- [ ] Update documentation (tracker env reference, runbook) and link the new Gate-focused behaviour; note any operational toggles or rollback steps.
- [ ] Capture validation evidence (workflow run logs, issue updates, label changes) for review before merging implementation PRs.
