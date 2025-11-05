# Issues 3260 & 3261 — Keepalive Workflow Tracker

_A consolidated evidence log for the keepalive poster (Issue #3260) and detector/dispatcher hardening (Issue #3261). Status tables remain evidence-first: no item flips to “complete” without a linked workflow run, log excerpt, or PR artifact._

---

## Issue 3260 — Keepalive Poster Enhancements Progress

### Task Tracking

| Task | Status | Verification Notes |
| --- | --- | --- |
| Helper module exports `makeTrace` and `renderInstruction`. | Complete | `.github/scripts/keepalive_contract.js` normalises inputs and prefixes the required hidden markers. |
| Orchestrator computes round/trace, selects token, posts comment via helper. | Complete | `Prepare keepalive instruction` job in `.github/workflows/agents-70-orchestrator.yml` resolves round/trace, chooses PAT, and renders the comment body. |
| Summary records round, trace, author, comment ID. | Complete | `Summarise keepalive instruction` step writes all four fields to `$GITHUB_STEP_SUMMARY`. |
| Reaction ack loop with 👀/🚀 handling. | Complete | `Ack keepalive instruction` adds 👀 then polls for 🚀 for 60 s at 5 s cadence. |
| Fallback dispatch and PR comment when ack missing. | Complete | Fallback steps emit the repository_dispatch payload and a one-line PR comment when acknowledgement fails. |

### Acceptance Criteria Tracking

| Acceptance Criterion | Status | Evidence |
| --- | --- | --- |
| New instruction comment created each cycle with required markers and @codex. | ❌ Not satisfied | Latest orchestrator runs [#19087371981](https://github.com/stranske/Trend_Model_Project/actions/runs/19087371981) and [#19087550223](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550223) ended before the first job, so no instruction comments were produced. |
| Comment author resolves to `stranske` (ACTIONS_BOT_PAT) or `stranske-automation-bot` fallback. | ❌ Not satisfied | No instruction comment was created in recent runs; author provenance remains unverified. |
| PR-meta ack observed or fallback dispatch + comment emitted. | ❌ Not satisfied | Ack loop never executed; orchestrator aborted pre-job and emitted neither reactions nor fallback comment. |
| Step summary includes Round, Trace, Author, CommentId. | ❌ Not satisfied | Run [#19087550223](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550223) produced no step summary. |

### Notes & Local Validation

- 2025-11-04 – Ran `PYTHONPATH=./src pytest tests/test_keepalive_workflow.py` (12 passed) to confirm helper + keepalive workflow coverage.
- 2025-11-04 – Ran `PYTHONPATH=./src pytest tests/test_workflow_agents_consolidation.py` (39 passed) to validate orchestration + PR-meta integration.
- 2025-11-04 – Ran `PYTHONPATH=./src pytest tests/test_workflow_naming.py` (7 passed) to ensure workflow naming conventions remain aligned.
- 2025-11-04 – Ran `PYTHONPATH=./src pytest tests/test_workflow_autofix_guard.py` (3 passed) to verify autofix guard workflow behaviour.
- 2025-11-04 – Ran `PYTHONPATH=./src pytest tests/test_workflow_multi_failure.py` (1 passed) to confirm multi-failure handling.
- 2025-11-05 – Guarded `workflow_dispatch` concurrency inputs to prevent push-triggered runs from failing before job execution; awaiting new detector/orchestrator cycle for validation.
- 2025-11-05 – Added detector tolerance for sanitized keepalive markers and expanded coverage via `tests/test_agents_pr_meta_keepalive.py` (5 passed locally); awaiting merged workflow run to consume the fix.

---

## Issue 3261 — Keepalive Detection & Dispatch Hardening Log

### Acceptance Criteria Status

| Acceptance Criterion | Status | Latest Evidence |
| --- | --- | --- |
| Valid instruction comment (allowed author, hidden markers) triggers PR-meta run reporting `ok: true`, `reason: keepalive-detected`, and populated round/trace/PR fields. | ❌ Not satisfied | `Agents PR meta manager` runs [#19087372965](https://github.com/stranske/Trend_Model_Project/actions/runs/19087372965) and [#19087550353](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550353) skipped detection, leaving fields unset. |
| Exactly one orchestrator `workflow_dispatch` fires with matching TRACE and no cancellations from other rounds. | ❌ Not satisfied | Orchestrator run [#19087550223](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550223) terminated pre-job; no TRACE propagation occurred. |
| Exactly one `codex-pr-comment-command` repository_dispatch emitted per accepted instruction comment. | ❌ Not satisfied | Detector run [#19087550353](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550353) never reached dispatch; no events recorded. |
| Guard failures yield PR comment `Keepalive {round} {trace} skipped: <reason>` plus matching summary entry. | ❌ Not satisfied | Orchestrator abort prevented skip comment + summary emission. |
| Two consecutive valid rounds produce distinct traces, distinct orchestrator runs, and no duplicate dispatches. | ❌ Not satisfied | No qualifying consecutive rounds yet. |

### Task List Status

| Task Group | Task | Status | Latest Evidence |
| --- | --- | --- | --- |
| PR-meta detector | Ensure `actions/checkout@v4` occurs before loading detector script. | ⏳ In progress | Checkout step present in `.github/workflows/agents-pr-meta.yml` (commit `777ce17d`); awaiting runtime confirmation. |
| PR-meta detector | Enforce allowed authors + hidden markers, surface structured outputs. | ⏳ In progress | `.github/scripts/agents_pr_meta_keepalive.js` emits comment ID/URL and `missing-round`; need live run to confirm outputs. |
| PR-meta detector | Add 🚀 dedupe and dispatch orchestrator (`workflow_dispatch`) + connector (`repository_dispatch`). | ⏳ In progress | Repository dispatch now forwards round/trace plus fallback metadata (commit `777ce17d`); validation pending next accepted keepalive. |
| PR-meta detector | Emit summary table on every run. | ⏳ In progress | Summary step updated with Comment column; awaiting detector evidence. |
| Orchestrator | Parse `options_json`, export TRACE/ROUND/PR, configure `concurrency` without cancel-in-progress. | ⏳ In progress | Parameter resolver populates keepalive metadata; need next orchestrator run to confirm. |
| Orchestrator | Post `Keepalive … skipped:` PR comment + summary when guard fails. | ⏳ In progress | Guard posts formatted skip line (commit `2ce66b54`); awaiting skip event evidence. |
| Orchestrator | Filter assignees to humans; skip gracefully when none remain. | ⏳ In progress | Guard auto-assigns humans and applies `agents:keepalive`/`agent:codex` labels (commit `e3dc4c65`); confirmation pending live run. |

### Evidence Log

| Timestamp (UTC) | Event | Notes |
| --- | --- | --- |
| 2025-11-02 09:14 | Issue synced by workflow run [#19060644912](https://github.com/stranske/Trend_Model_Project/actions/runs/19060644912) | Baseline import from topic GUID `c99d3476-9806-5144-8a69-98a586644cbd`. No compliant runs yet. |
| 2025-11-04 17:59 | Gate workflow run [#19078172801](https://github.com/stranske/Trend_Model_Project/actions/runs/19078172801) | Shows current failure mode (coverage test AttributeError); lacks TRACE propagation. |
| 2025-11-04 22:24 | Orchestrator run [#19084601666](https://github.com/stranske/Trend_Model_Project/actions/runs/19084601666) | Workflow ended before job steps; no TRACE export or skip comment. |
| 2025-11-04 22:25 | Agents PR meta manager run [#19084629319](https://github.com/stranske/Trend_Model_Project/actions/runs/19084629319) | Detection table `ok=false`, `reason=not-keepalive`; dispatch hooks dormant awaiting hidden markers. |
| 2025-11-04 23:08 | Updated orchestrator idle precheck for explicit keepalive dispatches. | Should allow detector-triggered runs to reach keepalive jobs even without open agent issues; validation pending. |
| 2025-11-04 23:16 | Normalised keepalive skip comment format in orchestrator guard. | Guarantees `Keepalive {round} {trace} skipped: <reason>` comment + summary; awaiting skip event. |
| 2025-11-04 23:24 | Added comment metadata + specific missing-round reason to detector. | Detector now emits comment ID/URL and differentiates missing round markers; summary table gains comment column. Awaiting live run. |
| 2025-11-04 23:31 | Forwarded keepalive round/trace with repository dispatch. | Codex dispatch attaches round/trace and falls back to detector comment metadata; awaiting valid keepalive for evidence. |
| 2025-11-04 23:38 | Keepalive guard auto-labels PR before failing. | Guard attempts to apply `agents:keepalive` / `agent:codex` labels before logging skip; validate on next label-missing case. |
| 2025-11-04 23:48 | Added harness + pytest coverage for detector metadata + missing-round reason. | `tests/test_agents_pr_meta_keepalive.py` asserts new outputs using fixtures under `tests/fixtures/agents_pr_meta/`. |
| 2025-11-05 00:38 | Agents PR meta manager run [#19087372965](https://github.com/stranske/Trend_Model_Project/actions/runs/19087372965) | Detection/dispatch jobs skipped; no summary or metadata emitted. |
| 2025-11-05 00:48 | Agents PR meta manager run [#19087550353](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550353) | Push-triggered run ended before jobs; detector evidence still pending. |
| 2025-11-05 00:49 | Orchestrator run [#19087550223](https://github.com/stranske/Trend_Model_Project/actions/runs/19087550223) | Workflow terminated at setup; skip comment & summary unverified. |
| 2025-11-05 01:05 | Guarded `workflow_dispatch` concurrency inputs. | `.github/workflows/agents-70-orchestrator.yml` now short-circuits missing inputs to prevent push-triggered job failures. Awaiting next detector/orchestrator cycle. |

### Upcoming Verification Steps

1. Capture the next detector run following a keepalive instruction comment with hidden markers from an allowed author; record detector outputs showing `dispatch=true`, `keepalive-detected`, and populated metadata.
2. Confirm the orchestrator run triggered by that comment, including TRACE/ROUND propagation, summary output, and ack/fallback handling.
3. Observe exactly one `codex-pr-comment-command` repository dispatch for the accepted instruction comment.
4. Intentionally trigger a guard failure to verify the `Keepalive {round} {trace} skipped:` PR comment and summary entry.
5. Run two valid consecutive rounds to ensure distinct traces, no duplicate dispatches, and absence of cancellations.
