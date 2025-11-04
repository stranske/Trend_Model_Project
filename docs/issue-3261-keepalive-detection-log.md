# Issue 3261 – Keepalive Detection & Dispatch Hardening Log

_Evidence-first tracker for Issue #3261 (keepalive PR-meta detector and orchestrator hardening). Mirrors the Issue #3260 workflow: every status reflects verifiable GitHub Actions runs or PR artefacts so regressions surface immediately._

## Working Process

1. **Evidence precedes status** – No task or acceptance criterion is marked complete until the associated workflow run, log segment, or PR artefact is linked.
2. **Regression guard** – When fresh evidence lands, re-confirm all previously satisfied items under the latest production context. Downgrade status immediately if a check regresses.
3. **Author + marker focus** – Verification requires both an allowed author (`stranske` via ACTIONS_BOT_PAT or `stranske-automation-bot`) and the hidden keepalive markers before claiming success.
4. **Trace continuity** – Every round must yield consistent TRACE propagation across detector, orchestrator, and comment outputs; mismatches force investigation before closure.
5. **Audit trail** – Evidence Log entries are append-only with timestamps, run IDs, and direct links/quotes. Earlier entries remain for historical comparison.

## Acceptance Criteria Status

| Acceptance Criterion | Status | Latest Evidence |
| --- | --- | --- |
| Valid instruction comment (allowed author, hidden markers) triggers PR-meta run reporting `ok: true`, `reason: keepalive-detected`, and populated round/trace/PR fields. | ❌ Not satisfied | No post-issue detector run with hidden markers yet; most recent keepalive detection evidence predates Issue #3261 requirements. |
| Exactly one orchestrator workflow_dispatch fires with matching TRACE and no cancellations triggered by other keepalive rounds. | ❌ Not satisfied | Orchestrator runs following Issue #3261 creation have either been cancelled pre-job or have not carried TRACE markers; waiting for compliant cycle. |
| Exactly one repository_dispatch (`codex-pr-comment-command`) emitted per accepted instruction comment. | ❌ Not satisfied | No recent repository_dispatch payload observed in Gate summaries after Issue #3261 sync run [#19060644912](https://github.com/stranske/Trend_Model_Project/actions/runs/19060644912). |
| Guard failures yield one-line PR comment `Keepalive {round} {trace} skipped: <reason>` and matching summary entry. | ❌ Not satisfied | Existing guard failures (e.g., orchestrator cancellations on Issue #3260) did not post the explicit comment or summary rows mandated here. |
| Two consecutive valid rounds produce distinct traces, distinct orchestrator runs, and no duplicate dispatches. | ❌ Not satisfied | No consecutive compliant rounds recorded since the issue sync; pending first successful validation run. |

## Task List Status

| Task Group | Task | Status | Latest Evidence |
| --- | --- | --- | --- |
| PR-meta detector | Ensure `actions/checkout@v4` occurs before loading detector script. | ❌ Not satisfied | Current workflow definition not yet reviewed in this context; awaiting evidence from upcoming PR. |
| PR-meta detector | Enforce allowed authors + hidden markers, surface structured outputs. | ❌ Not satisfied | Detector logs still report missing marker handling; no run showing new parsing logic. |
| PR-meta detector | Add 🚀 dedupe and dispatch both orchestrator (`workflow_dispatch`) and connector (`repository_dispatch`) with contextual payload. | ❌ Not satisfied | No workflow logs referencing the new dispatch calls. |
| PR-meta detector | Emit summary table on every run. | ❌ Not satisfied | Latest detector runs terminate early without summary output segment. |
| Orchestrator | Parse `options_json`, export TRACE/ROUND/PR, and configure `concurrency` without cancel-in-progress. | ❌ Not satisfied | Modern orchestrator invocations still cancel on overlap; concurrency block unchanged. |
| Orchestrator | Before bailing, post `Keepalive … skipped:` PR comment and mirror reason in `$GITHUB_STEP_SUMMARY`. | ❌ Not satisfied | Failure handling currently silent; no PR comments or summaries citing reason codes. |
| Orchestrator | Filter assignees to humans; skip gracefully when none remain. | ❌ Not satisfied | Assignee enforcement still fails closed when filtered list is empty; no evidence of new guard. |

## Evidence Log

| Timestamp (UTC) | Event | Notes |
| --- | --- | --- |
| 2025-11-02 09:14 | Issue synced by workflow run [#19060644912](https://github.com/stranske/Trend_Model_Project/actions/runs/19060644912) | Issue body imported from topic GUID `c99d3476-9806-5144-8a69-98a586644cbd`. Serves as baseline; no compliant detector/orchestrator runs recorded yet. |
| 2025-11-04 17:59 | Gate workflow run [#19078172801](https://github.com/stranske/Trend_Model_Project/actions/runs/19078172801) | Demonstrates current failure mode (AttributeError in coverage tests) but lacks TRACE propagation or skip-comment output required by Issue #3261. |

## Next Verification Steps

1. **Capture next detector run** – Wait for a keepalive instruction comment carrying both hidden markers from an allowed author. Record detector run outputs verifying new parsing, dedupe, and dispatch behaviour.
2. **Confirm orchestrator wiring** – For the same round, document the dispatched workflow run ID, ensure `concurrency.cancel-in-progress: false`, and capture the TRACE/ROUND environment lines.
3. **Repository dispatch proof** – Pull API logs or run summaries confirming exactly one `codex-pr-comment-command` dispatch per accepted instruction comment.
4. **Skip-path validation** – Intentionally trip a guard (e.g., missing label) once changes land to confirm the orchestrator posts the mandated one-line skip comment and summary row.
5. **Double-round regression** – After first pass succeeds, trigger a second valid instruction to verify unique traces, non-cancelling runs, and absence of duplicate comments.

_All future edits must preserve historical entries in the Evidence Log and re-evaluate earlier statuses whenever new evidence is introduced._
