# Failure Tracker Workflow Scope Plan

## Scope and Key Constraints
- Limit workflow runs to `workflow_run` events originating from pull request workflows; ignore pushes, scheduled, or manual triggers.
- Operate only on repositories where the Gate follower (`maint-30-post-ci.yml`) is configured; legacy `maint-33-check-failure-tracker.yml` shells simply delegate.
- Apply labeling logic solely to pull requests with failed required checks to avoid interfering with optional jobs.
- Produce exactly one small `ci-failures-snapshot` artifact per failing workflow run (and the same payload on the auto-heal success path); keep artifact size minimal (<1 MB) and avoid storing secrets or PII.
- Maintain compatibility with existing success-path behaviour so passing runs remain label/artifact free.

## Acceptance Criteria / Definition of Done
1. **Trigger Filtering** – The failure tracker only evaluates runs whose source workflow was triggered by a pull request event.
2. **Label Management** – When any required job fails, the owning pull request is labeled `ci-failure` (label is added once, no duplicates, removed when checks later pass).
3. **Artifact Handling** – Each failing run uploads a single lightweight `ci-failures-snapshot` artifact containing the expected summary payload; no extra artifacts are emitted.
4. **Idempotent Behaviour** – Re-running the workflow on the same failing run maintains a single label and artifact without accumulating duplicates.
5. **Documentation Update** – Repository docs explain the tightened scope, labeling behaviour, artifact expectations, and maintenance notes for future operators.
6. **Verification Evidence** – Provide replay or dry-run notes showing at least one failing and one passing scenario, including label and artifact results.

## Initial Task Checklist
- [x] Audit the historical `maint-33-check-failure-tracker.yml` triggers and add guard clauses to ensure the job exits early unless the source event is a PR.
- [x] Implement or refine logic that detects failed required jobs and adds/removes the `ci-failure` label accordingly inside `maint-30-post-ci.yml`.
- [x] Confirm the artifact upload step reuses the `ci-failures-snapshot` name and enforce size/content constraints.
- [x] Update documentation (this plan plus any operator guides) to capture scope, label expectations, and artifact lifecycle.
- [x] Validate behaviour via `workflow_run` replays or local action runners, capturing evidence for both passing and failing PR scenarios.

## Verification Notes
- **Failing run replay** – Re-run a failing PR workflow and confirm the failure tracker job emits a single `ci-failures-snapshot` artifact and applies (or retains) a lone `ci-failure` label on the PR timeline.
- **Passing run replay** – Trigger a successful re-run for the same PR and confirm the success job removes the `ci-failure` label and no additional artifacts are emitted beyond the summary snapshot.
- **Label hygiene** – Inspect the PR event log to ensure the label add/remove operations occur only once per run, matching the acceptance criteria for singular labeling.
- **Artifact audit** – Download the `ci-failures-snapshot` artifact from both scenarios to verify the payload contains only the expected JSON keys and remains under the size threshold.
- **Automated regression** – `pytest tests/test_failure_tracker_workflow_scope.py` validates the workflow definition enforces PR-only execution, single-label semantics, and one snapshot artifact per failing run, providing local evidence that mirrors the replay expectations.
