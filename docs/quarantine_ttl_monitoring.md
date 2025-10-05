# Quarantine TTL Monitoring (maint-34)

> **Status:** Workflow retired during Issue #2190. The notes below remain for
> historical context should a replacement hygiene check be reintroduced.

`maint-34-quarantine-ttl.yml` enforces the "quarantine entries must expire"
policy by linting `tests/quarantine.yml` on a schedule and during relevant PRs.
It uses the shared validator `tools/validate_quarantine_ttl.py` so automation
and maintainers get the same behaviour locally.

## Trigger surface

- Nightly cron at 04:30 UTC.
- Manual `workflow_dispatch` for spot checks.
- Targeted PR/push triggers scoped to `tests/quarantine.yml`, the validator, and
  the workflow definition.

## Step summary example

```
## Quarantine TTL validation

- Total entries scanned: 1
- ✅ No expired quarantines detected.
```

Failures list each offending test ID alongside its expiry (or parsing error),
which is the same payload surfaced in the gate orchestrator job.

## Gate integration

`pr-01-gate-orchestrator.yml` job `orchestrator / quarantine-ttl` imports the same
validator so expired entries fail the PR gate immediately. The gate job emits a
message such as:

```
Expired or invalid quarantine entries detected
- tests/test_alpha.py::test_expired (expired 2024-05-01)
```

Keep the validator's error strings informative; the gate output is the only
context reviewers see when the check fails.

## Local smoke test

Run the lightweight harness to verify the validator without waiting for Actions:

```bash
python scripts/workflow_smoke_tests.py
```

The script materialises a temporary quarantine file, runs the validator, and
prints the same Markdown summary shown in workflow logs.
