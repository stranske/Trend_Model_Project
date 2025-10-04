# Maintenance Workflow Playbook

This playbook explains how on-call responders should handle failures in the
maint-3x maintenance series introduced for Issue 1664.

## maint-34-quarantine-ttl.yml

1. **Check the job summary** — the step summary lists each expired ID and its
   expiry date.
2. **Decide the fix** — either update the underlying test so it can graduate
   from quarantine or extend the TTL with a follow-up issue reference.
3. **Verify locally** — run `python tools/validate_quarantine_ttl.py` (or the
   smoke harness `python scripts/workflow_smoke_tests.py`) before re-pushing.
4. **Gate impact** — the gate orchestrator fails the PR while expired entries
   remain, so acknowledge the check in the PR discussion if additional work is
   required.

## maint-35-repo-health-self-check.yml

1. **Read the Ops issue update** — failures append to the canonical issue marked
   `<!-- repo-health-nightly -->`. If `OPS_HEALTH_ISSUE` is unset the workflow
   will warn in the summary instead.
2. **Diagnose quickly** — the summary highlights missing labels, secrets, or
   workflow lint failures. Re-run the workflow via `workflow_dispatch` after
   remediation to confirm the fix.
3. **Update governance assets** — add the missing label/secret/variable via the
   repository settings UI. For actionlint failures, follow the inline reviewdog
   comments on the relevant PR.

## maint-36-actionlint.yml

1. **PR context** — reviewdog publishes annotations directly on the offending
   workflow file. Open the "Files changed" tab to see inline comments.
2. **Scheduled run** — when the weekly cron fails, inspect the workflow logs to
   identify the offending file and push a corrective PR. The gate orchestrator
   runs the same lint, so broken syntax will also block merges.
3. **Version drift** — actionlint is pinned; if you need a newer version, update
   both `maint-36-actionlint.yml` and `pr-18-gate-orchestrator.yml` in the same PR and
   record the change in `docs/workflow`.

## Common tooling

- `python scripts/workflow_smoke_tests.py` exercises the repo-health probe and
  quarantine TTL validator offline.
- All three workflows append Markdown to `$GITHUB_STEP_SUMMARY`, making it easy
  to snapshot their state in dashboards or incident notes.
