# Maintenance Workflow Playbook

This playbook explains how on-call responders should handle failures in the
maintenance workflows that remain after Issue 2190. The roster now consists of
`maint-02`, `maint-30`, `maint-32`, `maint-33`, `maint-36`, `maint-40`, and
`maint-90`.

## maint-02-repo-health.yml

1. **Open the run summary** — the weekly sweep writes a single report that
   lists stale branches (older than the configured threshold) and open issues
   without assignees. The heading `Repository health weekly sweep` marks the
   latest results.
2. **Handle stale branches** — follow the table entries to prune abandoned
   branches or push a fresh commit if the branch is still active. The table is
   capped at 20 rows; the summary includes an overflow note when additional
   branches are hidden.
3. **Triage unassigned issues** — assign an owner or update labels for the
   surfaced issues so they no longer show up in the next run. Issues are sorted
   by oldest activity first.
4. **Adjust the threshold if needed** — update the repository variable
   `REPO_HEALTH_STALE_BRANCH_DAYS` when the stale-branch window should change.
   Re-run the workflow via `workflow_dispatch` to validate the new threshold.

## maint-30-post-ci-summary.yml

1. **Check the run summary** — each execution appends a consolidated status
   block under `## Automated Status Summary`. Read it to understand which jobs
   failed.
2. **Re-run after fixes** — once the underlying CI run passes, manually re-run
   the workflow (or re-run the source CI) to refresh the run summary entry.
3. **Keep paths aligned** — if this workflow starts failing to locate artifacts
   ensure `pr-10-ci-python.yml` and `pr-12-docker-smoke.yml` still expose the
   expected outputs.

## maint-32-autofix.yml

1. **Inspect the uploaded patch** — on failure the workflow attaches the patch
   bundle to the run. Download it and review the diff locally.
2. **Mind the exit conditions** — the job fails only when the autofix could not
   be pushed automatically. Apply the patch locally, re-run formatting, and push
   manually.
3. **Update labels if needed** — the workflow relies on PR labels to decide
   whether autofix should attempt a commit. Confirm the labels are present when
   debugging skipped runs.

## maint-33-check-failure-tracker.yml

1. **Review the issue comment** — failures create or update the CI failure
   tracker issue with signature details. Follow the links to the failing jobs.
2. **Respect cooldowns** — the workflow enforces rate limits to prevent spam.
   When testing fixes, wait for the cooldown or adjust the environment variables
   in a fork.
3. **Confirm auto-heal** — after the offending workflow run passes, trigger the
   tracker again to ensure the issue closes (or the comment updates to show the
   heal state).

## maint-36-actionlint.yml

1. **PR context** — reviewdog publishes annotations directly on the offending
   workflow file. Open the "Files changed" tab to see inline comments.
2. **Scheduled run** — when the weekly cron fails, inspect the workflow logs to
   identify the offending file and push a corrective PR. Actionlint failures
   will also block merges via the CI gate.
3. **Version drift** — actionlint is pinned; if you need a newer version, update
   the pinned release in both this workflow and the reusable composites that run
   actionlint.

## maint-40-ci-signature-guard.yml

1. **Check signature metadata** — the summary highlights which manifest failed
   verification. Validate the signature locally using the instructions in
   `docs/ci-signature-guard.md`.
2. **Re-issue manifests** — regenerate the signed manifest or rotate the key
   material as required. Ensure the updated signature is committed before
   re-running the workflow.
3. **Watch branch filters** — this workflow only runs on `phase-2-dev` and
   branches prefixed `feat/` or `chore/`. If it appears missing, confirm the
   branch naming matches the allow list.

## maint-90-selftest.yml

1. **Treat it as a smoke harness** — the workflow invokes
   `reusable-99-selftest.yml` to exercise the reusable CI matrix.
2. **Verify inputs** — ensure the forwarded `python-version` input matches the
   expected matrix before debugging downstream failures.
3. **Secrets passthrough** — the workflow forwards secrets to the reusable call.
   When adding new secrets to the reusable workflow remember to surface them
   here as well.

## Common tooling

- All workflows append Markdown to `$GITHUB_STEP_SUMMARY`, making it easy
  to snapshot their state in dashboards or incident notes.
