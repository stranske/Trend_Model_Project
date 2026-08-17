# Keepalive Monitoring Checklist

Use this checklist to validate the current consolidated keepalive path. The consumer entry points are
`.github/workflows/agents-81-gate-followups.yml` for Gate-triggered or manual follow-ups and
`.github/workflows/agents-keepalive-sweep.yml` for scheduled recovery. Generated consumer workflow
changes must come from the canonical `stranske/Workflows` source and its sync process.

## 1. Select an active agent PR

```bash
gh pr list --label "agents:keepalive" --state open --limit 5
gh pr view PRNUM --json headRefOid,isDraft,labels,statusCheckRollup
```

The PR must be open and ready for review. Confirm it has one supported concrete route,
`agent:codex` or `agent:claude`; the current Gate-followup workflow has no Cursor or Gemini runner
job.

## 2. Inspect the current follow-up runs

```bash
gh run list --workflow agents-81-gate-followups.yml --limit 10
gh run list --workflow agents-keepalive-sweep.yml --limit 10
```

For a specific run, inspect the evaluation and runner jobs:

```bash
gh run view RUN_ID --json headSha,status,conclusion,jobs
gh run view RUN_ID --log
```

The evaluated head must match the PR head. A successful evaluation passes the task appendix directly
to the selected shared runner; the current wrapper does not publish a task-appendix artifact.

## 3. Inspect durable evidence

The Gate-followup summary job publishes a `keepalive-metrics` artifact. Download and inspect it:

```bash
gh run download RUN_ID --name keepalive-metrics --dir /tmp/keepalive-metrics-RUN_ID
jq . /tmp/keepalive-metrics-RUN_ID/keepalive-metrics.ndjson
```

Confirm the PR number, action, stop reason, task totals, and Gate conclusion match the PR's current
state. Then check the PR's status comment and work log for the same iteration and head.

## 4. Force one bounded retry when appropriate

The `agent:retry` label is an observability marker; applying it alone does not set the workflow's
`force_retry` input. Use an explicit dispatch:

```bash
gh workflow run agents-81-gate-followups.yml \
  -f pr_number=PRNUM \
  -f force_retry=true
```

Confirm the dispatched run before removing stale `agent:retry` or `agent:rate-limited` labels. Do not
use forced retries to bypass a human-owned blocker, an unsupported runner, or a changed PR head.

## 5. Completion checks

- The run evaluated the unchanged PR head.
- The selected runner was Codex or Claude and actually ran, or the summary records a concrete stop
  reason.
- `keepalive-metrics` agrees with the PR status comment and current task counts.
- Changed files remain within the PR's stated scope.
- Recovery labels do not remain after the recovered state is confirmed.
- Required checks, review threads, and merge state are checked separately before merge.
