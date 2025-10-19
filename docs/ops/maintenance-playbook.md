# Maintenance Workflow Playbook

This playbook explains how on-call responders should handle failures in the
maintenance workflows that remain after Issue 2190. The roster now consists of
`health-41`, `maint-46-post-ci`, `health-42`, `health-43`,
`health-44`, `maint-45-cosmetic-repair`, and `maint-90`.

## health-41-repo-health.yml

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

## maint-46-post-ci.yml

1. **Check the run summary** — each execution appends a consolidated status
   block under `## Automated Status Summary`. Review it to understand which jobs
   failed and whether autofix attempted a patch.
2. **Inspect the uploaded patch** — when autofix cannot push directly the
   workflow uploads a patch artifact. Download it, apply locally, re-run
   formatting, and push manually if appropriate.
3. **Re-run after fixes** — once underlying CI passes (or the patch is applied)
   manually re-run the workflow to refresh the summary and confirm autofix
   reports success.
4. **Keep inputs aligned** — if artifact lookups fail, ensure `pr-00-gate.yml`
   still exposes the expected coverage/smoke outputs and that required labels
   remain in place for autofix eligibility.
5. **Monitor the CI failure tracker** — the `failure-tracker` job creates or
   updates the consolidated CI failure issue when Gate reports failures and
   automatically resolves it once the offending run passes. Follow the links in
   the run summary to triage recurring signatures.

## health-42-actionlint.yml

1. **PR context** — reviewdog publishes annotations directly on the offending
   workflow file. Open the "Files changed" tab to see inline comments.
2. **Scheduled run** — when the weekly cron fails, inspect the workflow logs to
   identify the offending file and push a corrective PR. Actionlint failures
   will also block merges via the CI gate.
3. **Version drift** — actionlint is pinned; if you need a newer version, update
   the pinned release in both this workflow and the reusable composites that run
   actionlint.

## health-43-ci-signature-guard.yml

1. **Check signature metadata** — the summary highlights which manifest failed
   verification. Validate the signature locally using the instructions in
   `docs/ci-signature-guard.md`.
2. **Re-issue manifests** — regenerate the signed manifest or rotate the key
   material as required. Ensure the updated signature is committed before
   re-running the workflow.
3. **Watch branch filters** — this workflow only runs on `phase-2-dev` and
   branches prefixed `feat/` or `chore/`. If it appears missing, confirm the
   branch naming matches the allow list.

## health-44-gate-branch-protection.yml

1. **PAT availability** — ensure the fine-grained `BRANCH_PROTECTION_TOKEN`
   secret is configured with branch-protection admin scope. When absent the
   workflow exits early after logging a warning; configure the secret before
   re-running.
2. **Review summary output** — the run summary lists any missing branch
   protection rules alongside the enforced baseline from
   `tools/enforce_gate_branch_protection.py`. Apply the suggested fixes via the
   repository settings UI or API.
3. **Manual reruns** — after updating the rules, rerun the workflow to confirm
   the summary reports a clean state.

## maint-45-cosmetic-repair.yml

1. **Inspect pytest results** — the workflow records the pytest exit code in
   the job summary even though the step allows failures. Investigate test
   regressions before applying cosmetic fixes.
2. **Check repair output** — `scripts/ci_cosmetic_repair.py` prints the diff of
   any adjustments (tolerance bumps, snapshot rewrites). Review the summary to
   confirm the intended fixes before pushing.
3. **Dry-run mode** — use the `dry-run` input for rehearsal runs that capture
   diagnostics without creating commits or PRs.

## Manual self-test runner

1. **Pick the right mode** — dispatch `selftest-reusable-ci.yml` and choose `summary`,
   `comment`, or `dual-runtime` depending on whether you want a workflow
   summary-only run, a PR comment, or a dual Python matrix. Supply the optional
   `python_versions` override when you need interpreters outside the defaults.
2. **Provide required inputs** — comment mode requires a `pull_request_number`
   when `post_to` is `pr-number`. Optional overrides let you change comment and
   summary titles without editing the workflow.
3. **Review the embedded matrix** — verification mismatches cause the runner
   job to fail (unless the run is comment-only) after surfacing the Markdown
   summary/comment. Use `enable_history: true` when you need the JSON artifact
   for offline inspection.

## Common tooling

- All workflows append Markdown to `$GITHUB_STEP_SUMMARY`, making it easy
  to snapshot their state in dashboards or incident notes.
