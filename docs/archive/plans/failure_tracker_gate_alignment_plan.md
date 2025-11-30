# Failure Tracker Gate Integration Plan (Issue #2436)

## Scope & Key Constraints
- Repoint the failure-tracker automation to listen for `workflow_run` events emitted by the **Gate** workflow only; do not widen event scope beyond Gate or alter other workflow triggers.
- Consolidate tracker update logic into the Gate summary job within `pr-00-gate.yml` while keeping backward-compatible hooks for the legacy tracker workflow until removal is validated.
- Maintain single-source-of-truth updates for the failure tracker issue and PR comment; avoid introducing additional bot comment pathways.
- Preserve existing rate limits, cooldown handling, and deduplication semantics already defined in the tracker to prevent regression in issue/comment noise.
- Ensure CI label management remains idempotent and aligns with repository labeling conventions (e.g., create `ci-failure` on demand without duplicating other labels).
- Operate within GitHub Actions constraints (no new third-party actions without security review; reuse in-repo composite actions or patterns where possible).
- Provide observability (logs/artifacts) sufficient to validate Gate integration without enabling noisy verbose modes by default.

## Acceptance Criteria / Definition of Done
1. Failure tracker no longer reacts to runs from PR 10 or PR 12 (historical duplicates) once Gate rerouting is active.
2. On any qualifying Gate failure:
   - The Gate summary job orchestrates the tracker update sequence and produces exactly one updated issue entry and one persistent PR comment.
   - No additional bot comments appear on the PR from other workflows.
3. Successful application (and removal on recovery) of the `ci-failure` label via the consolidated workflow.
4. Legacy failure-tracker workflow is removed once delegation stability is confirmed, leaving the Gate summary job as the sole listener.
5. Telemetry/artifacts confirm the workflow path taken (Gate → Gate summary job → tracker update) for auditability.
6. Documentation/config references updated to reflect the new event source and workflow responsibilities.
7. Consolidated workflow publishes the same `ci-failures-snapshot` artifact for failure and success reconciliation paths.

## Initial Task Checklist
- [x] Audit the historical `.github/workflows/maint-47-check-failure-tracker.yml` shell to catalog tracker update logic, shared environment variables, and comment/issue handling prior to removal.
- [x] Update the failure-tracker workflow trigger to `on: workflow_run` for `workflows: ["Gate"]` with `types: [completed]`; ensure proper filtering for branch/PR scope.
- [x] Refactor tracker update steps into the Gate summary job, preserving inputs, secrets, and retry/backoff behavior; leave a thin shell in the original workflow if needed for compatibility.
- [x] Wire the Gate summary job to invoke the tracker update path conditionally based on Gate outcomes (failed, cancelled, neutral).
- [x] Implement `ci-failure` label management in the Gate summary job (apply on failure, remove when passing) with existing helper scripts or GitHub CLI usage.
- [x] Verify that duplicate comment suppression still functions when updates originate from the consolidated workflow (reuse existing markers/timestamps).
- [x] Exercise dry-run or sandbox runs referencing PRs 10 and 12 to confirm no new events fire from those historical workflows.
- [x] Update documentation (tracker env reference, runbook) and link the new Gate-focused behaviour; note any operational toggles or rollback steps.
- [x] Capture validation evidence (workflow run logs, issue updates, label changes) for review before merging implementation PRs.

## Implementation Summary
- Failure tracker issue/label management now lives in the new `failure-tracker` segment within the Gate summary job, preserving the signature hashing, cooldown, and escalation semantics from the legacy workflow.
- `maint-47-check-failure-tracker.yml` was retired after delegation proved stable; the Gate summary job now stands alone.
- Artifact handling is unified so both failure and success paths emit a single `ci-failures-snapshot` payload sourced from the consolidated workflow run.
- Historical duplicate PRs (#10 and #12) are explicitly marked for failure-tracker skip so Gate replays against them do not emit new issues or labels.
- Tests and docs were updated to reflect the delegated architecture and to ensure future changes keep the Gate-only trigger contract intact.

## Validation Evidence
- Automated regression suite exercises the consolidated failure-tracker path (`pytest tests/test_failure_tracker_workflow_scope.py tests/test_workflow_naming.py`).
- The Gate follower exposes `failure_tracker_skip` for PRs #10/#12 and the Gate summary failure-tracker path requires the skip flag to be false before updating issues/labels.
- The Gate summary job publishes a single `ci-failures-snapshot` artifact in both failure and success runs and is the exclusive writer of the consolidated status comment, preventing duplicate bot posts.
