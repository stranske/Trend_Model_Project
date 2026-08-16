# Keepalive Goals and Plumbing

This is the canonical consumer-repository contract for keepalive. The executable sources of truth
are `.github/scripts/keepalive_loop.js`, `.github/scripts/keepalive_gate.js`, and the generated
workflow wrappers listed below. Change generated wrappers in `stranske/Workflows`, then deliver them
through the sync workflow; do not create a second local implementation.

## Purpose

Keepalive gives an already-open agent pull request bounded, repeatable follow-up work. It must keep
automation ownership through recoverable failures, stop at explicit operator or human-authority
boundaries, and preserve enough durable evidence to explain every dispatch or stop.

Keepalive does not decide whether a pull request is reviewed or mergeable. Review resolution,
required checks, exact-head stability, and merge authorization remain separate gates.

## Current entry points

| Surface | Responsibility |
| --- | --- |
| `.github/workflows/agents-80-pr-event-hub.yml` | Consolidates PR/comment/Gate events and calls the shared PR-meta implementation |
| `.github/workflows/agents-81-gate-followups.yml` | Evaluates state, dispatches the supported runner, persists the summary, and emits metrics |
| `.github/workflows/agents-keepalive-sweep.yml` | Hourly level-based recheck for non-draft open PRs carrying an `agent:*` label |
| `.github/scripts/keepalive_loop.js` | Canonical state machine, task accounting, runner selection, recovery, and durable state |
| `.github/scripts/keepalive_gate.js` | Activation, pause, label, and run-cap guardrails |

Agents 81 currently has runner jobs only for `agent:codex` and `agent:claude`. Registry entries for
other agents do not by themselves create a consumer Gate-followup runner.

## Lifecycle

1. Agents 80 handles qualifying PR, comment, and successful Gate events and delegates PR metadata to
   `reusable-20-pr-meta.yml` in the Workflows repository.
2. Agents 81 wakes after Gate, on its supported label event, or by explicit workflow dispatch. It
   evaluates the PR's current head and keepalive state through `keepalive_loop.js`.
3. The evaluator derives the next bounded action and passes the current task appendix directly to
   the Codex or Claude shared runner. The wrapper does not publish a task-appendix artifact.
4. The summary job persists the trusted keepalive state/work log and uploads the
   `keepalive-metrics` artifact.
5. The hourly sweep re-dispatches Agents 81 for eligible, non-draft agent PRs. It makes no work
   decision itself; the state machine and debounce remain authoritative.

## Guardrails

- Keepalive requires an eligible agent route and enabled keepalive state.
- `agents:paused`, `needs-human`, and `agents:max-runs:0` are durable stops. The compatibility spelling
  `agents:pause` may still be read by the implementation, but operators must use `agents:paused`.
- Per-PR concurrency is non-cancelling. An unchanged state fingerprint is a no-op, not authority to
  create another recovery attempt.
- Draft PRs are not sweep candidates. Agent pull requests are created ready for review and remain
  ready throughout automated follow-up.
- Generated sync PRs and intentional delivery holds are owned by the delivery workflow, not by the
  consumer autofix path.
- Cursor and Gemini labels must not be described as runnable consumer keepalive routes until their
  actual Agents 81 jobs are delivered from Workflows.

## Tasks and completion

The evaluator reads the current PR body and trusted keepalive state, reconciles task checkboxes, and
selects one action such as run, fix, conflict recovery, or stop. The runner receives the resulting
task appendix as an input. After work, the summary records task totals, completion progress, Gate
result, stop reason, and whether the head changed.

Task completion ends agent dispatch; it does not merge the PR. A closer must still verify the
unchanged exact head, required checks, active review threads, and clean merge state.

## Recovery

- Ordinary Gate completions and the hourly sweep re-evaluate without bypassing guardrails.
- A bounded manual retry uses `agents-81-gate-followups.yml` with `pr_number=<PR>` and
  `force_retry=true`.
- Applying `agent:retry` alone does not set that input in the consumer wrapper. Treat it as a recovery
  marker and confirm a successful dispatch before removing stale recovery labels.
- The sweep may force a signed, due authority challenge. Without valid signed provenance, the
  downstream workflow fails closed or performs only an ordinary recheck.
- Do not use forced retry to override `needs-human`, an unsupported runner, a changed head, or a
  delivery hold.

## Operator validation

```bash
gh pr view PRNUM --json headRefOid,isDraft,labels,statusCheckRollup
gh run list --workflow agents-81-gate-followups.yml --limit 10
gh run list --workflow agents-keepalive-sweep.yml --limit 10
gh run view RUN_ID --json headSha,status,conclusion,jobs
gh run download RUN_ID --name keepalive-metrics --dir /tmp/keepalive-metrics-RUN_ID
jq . /tmp/keepalive-metrics-RUN_ID/keepalive-metrics.ndjson
```

The run head, PR state/work log, and metrics artifact must agree. Use
`NEXT_STEPS_MONITORING.md` for the full current checklist and
`docs/keepalive/Observability_Contract.md` for evidence requirements.
