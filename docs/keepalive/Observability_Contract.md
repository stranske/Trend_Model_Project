# Keepalive Observability Contract

This contract defines the evidence required to diagnose the current consolidated keepalive path. It
describes observable state; the executable contract remains in `.github/scripts/keepalive_loop.js`
and `.github/scripts/keepalive_gate.js`.

## Evidence surfaces

| Evidence | Source | Required use |
| --- | --- | --- |
| PR head and labels | Pull request API | Establish the exact evaluated head and operator controls |
| Gate result | PR checks and Agents 81 inputs | Establish the conclusion used by the evaluator |
| Keepalive summary/state comment | `agents-81-gate-followups.yml` summary job | Durable action, reason, iteration, task counts, and trace |
| Work log | `keepalive_loop.js` | Per-round runner result and head/task progress |
| `keepalive-metrics` artifact | Agents 81 summary job | Machine-readable PR number, action, stop reason, Gate conclusion, task totals, and duration |
| Workflow run summary/jobs | Agents 80, Agents 81, and sweep runs | Dispatch provenance and the job that ran or skipped |

The task appendix is passed directly from the evaluator to the selected shared runner. Do not expect
or require a task-appendix artifact from the current consumer wrapper.

## Provenance requirements

- Every diagnosis records the PR number and full head SHA.
- Evidence from a prior head is historical and cannot authorize a new dispatch or merge.
- A sweep run is only a wakeup. Its dispatch count does not prove that a runner executed.
- A queued or in-progress run is not a terminal result.
- When the evaluator stops, the recorded reason and durable ownership state must be reported; a
  recoverable automation stop is not silently converted into a human blocker.
- Signed authority-challenge inputs are bound to repository, PR, fingerprint, nonce, and sweep run.
  Missing or invalid provenance must fail closed.

## Expected current routes

- `.github/workflows/agents-80-pr-event-hub.yml` calls `reusable-20-pr-meta.yml` for PR-meta handling.
- `.github/workflows/agents-81-gate-followups.yml` evaluates through `keepalive_loop.js` and can dispatch Codex or Claude.
- `.github/workflows/agents-keepalive-sweep.yml` dispatches Agents 81 hourly and never selects a runner itself.
- Explicit forced recovery is a workflow dispatch of Agents 81 with `pr_number` and
  `force_retry=true`; applying `agent:retry` alone is not dispatch evidence.

## Diagnosis matrix

| Observation | Interpretation | Next evidence |
| --- | --- | --- |
| No Agents 81 run after Gate | Consolidated mode, event eligibility, or workflow dispatch may be missing | Inspect `USE_CONSOLIDATED_WORKFLOWS`, Agents 80/81 run history, and Gate PR linkage |
| Agents 81 ran but runner jobs skipped | Evaluator selected a stop/no-op or an unsupported route | Read evaluator outputs, summary reason, labels, and agent type |
| Runner completed with no head change | Work may be advisory, empty, or stalled | Compare work log, task counts, and the next sweep recheck |
| `agent:rate-limited` remains | Backoff marker is still present | Confirm a later successful explicit retry before cleanup |
| Metrics disagree with PR state | Evidence may be stale or from another head | Re-resolve the run head and use the latest exact-head artifact |
| PR is draft | It is outside the scheduled sweep contract | Restore ready-for-review state; automation must not end with a draft |

## Minimum closeout record

A keepalive status report includes:

- PR URL, ready/draft state, and exact head SHA;
- owner and active automation state;
- latest terminal Agents 81 result and concrete reason;
- task totals and head-change evidence from the matching metrics/work log;
- remaining human action only when an independently confirmed human-owned boundary exists; and
- the next bounded automation action when work remains.

Keepalive completion does not establish merge readiness. Required checks, exact-head review, active
non-outdated review threads, mergeability, and clean merge state must be verified separately.
