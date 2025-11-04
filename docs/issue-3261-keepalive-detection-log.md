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
| Valid instruction comment (allowed author, hidden markers) triggers PR-meta run reporting `ok: true`, `reason: keepalive-detected`, and populated round/trace/PR fields. | ❌ Not satisfied | Latest detector run `Agents PR meta manager` [#19084629319](https://github.com/stranske/Trend_Model_Project/actions/runs/19084629319) reported `ok = false`, `reason = not-keepalive`; code now maps future missing markers to `missing-round` and surfaces comment metadata, pending live validation. |
| Exactly one orchestrator workflow_dispatch fires with matching TRACE and no cancellations triggered by other keepalive rounds. | ❌ Not satisfied | Orchestrator workflow_dispatch [#19084601666](https://github.com/stranske/Trend_Model_Project/actions/runs/19084601666) ended before executing jobs, so no TRACE propagation or successful dispatch was recorded. |
| Exactly one repository_dispatch (`codex-pr-comment-command`) emitted per accepted instruction comment. | ❌ Not satisfied | Detector run [#19084629319](https://github.com/stranske/Trend_Model_Project/actions/runs/19084629319) skipped dispatching; Gate summaries still show no repository_dispatch payloads after the new runs. |
| Guard failures yield one-line PR comment `Keepalive {round} {trace} skipped: <reason>` and matching summary entry. | ❌ Not satisfied | Orchestrator run [#19084601666](https://github.com/stranske/Trend_Model_Project/actions/runs/19084601666) failed pre-job and produced neither the skip comment nor the summary rows required by the issue. |
| Two consecutive valid rounds produce distinct traces, distinct orchestrator runs, and no duplicate dispatches. | ❌ Not satisfied | Detector/orchestrator pair [#19084629319](https://github.com/stranske/Trend_Model_Project/actions/runs/19084629319) / [#19084601666](https://github.com/stranske/Trend_Model_Project/actions/runs/19084601666) both failed validation, so the consecutive-round requirement remains unmet. |

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
| 2025-11-04 22:24 | Orchestrator run [#19084601666](https://github.com/stranske/Trend_Model_Project/actions/runs/19084601666) | Workflow concluded before any job-level steps executed; no TRACE export, skip comment, or summary output captured. |
| 2025-11-04 22:25 | `Agents PR meta manager` run [#19084629319](https://github.com/stranske/Trend_Model_Project/actions/runs/19084629319) | Detection table showed `ok = false`, `reason = not-keepalive`; dispatch hooks remained dormant awaiting hidden markers and allowed author. |
| 2025-11-04 23:08 | Updated `.github/workflows/agents-70-orchestrator.yml` idle precheck to honour explicit keepalive dispatches | Should allow detector-triggered runs to reach keepalive jobs even when no additional agent issues are open; validation pending next run. |
| 2025-11-04 23:16 | Normalised keepalive skip comment format in orchestrator guard | Guarantees guard failures leave a `Keepalive {round} {trace} skipped: <reason>` comment plus matching summary; awaiting new skip event for validation. |
| 2025-11-04 23:24 | Added comment metadata + specific missing-round reason to detector | `.github/scripts/agents_pr_meta_keepalive.js` now emits comment ID/URL and differentiates missing round markers via `missing-round`; summary table gains a Comment column. Confirmation awaits next detector execution. |
| 2025-11-04 23:31 | Forwarded keepalive round/trace with repository dispatch | Codex dispatch now attaches round/trace and falls back to detector comment metadata, enabling trace continuity checks once a valid keepalive fires. Awaiting new dispatch for evidence. |
| 2025-11-04 23:38 | Keepalive guard auto-labels PR before failing | `keepalive-guard` attempts to apply `agents:keepalive` and `agent:codex` labels automatically before recording a skip reason; new behaviour to be validated when a label-missing scenario reoccurs. |
| 2025-11-04 23:48 | Added harness + pytest coverage for comment metadata + missing-round reason | `tests/test_agents_pr_meta_keepalive.py` now asserts the detector outputs comment ID/URL and returns `missing-round` when round markers are absent, using new fixtures under `tests/fixtures/agents_pr_meta/`. |

## Next Verification Steps

1. **Capture next detector run** – Wait for a keepalive instruction comment carrying both hidden markers from an allowed author. Record detector run outputs verifying new parsing, dedupe, and dispatch behaviour.
2. **Confirm orchestrator wiring** – For the same round, document the dispatched workflow run ID, ensure `concurrency.cancel-in-progress: false`, and capture the TRACE/ROUND environment lines.
3. **Repository dispatch proof** – Pull API logs or run summaries confirming exactly one `codex-pr-comment-command` dispatch per accepted instruction comment.
4. **Skip-path validation** – Intentionally trip a guard (e.g., missing label) once changes land to confirm the orchestrator posts the mandated one-line skip comment and summary row.
5. **Double-round regression** – After first pass succeeds, trigger a second valid instruction to verify unique traces, non-cancelling runs, and absence of duplicate comments.

_All future edits must preserve historical entries in the Evidence Log and re-evaluate earlier statuses whenever new evidence is introduced._
