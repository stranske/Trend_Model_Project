## Automated Post-CI Status Summary

The repository maintains a single, continuously updated pull-request comment
that surfaces CI and Docker results once both workflows finish. The
`maint-30-post-ci-summary.yml` follower reacts to `workflow_run` events for the
`CI` and `Docker` workflows, downloads shared artifacts, and renders a unified
Markdown block headed by `### Automated Status Summary`.

### Comment Contents

Each refresh includes:

* Head commit SHA and a roll-up of the most recent workflow runs that were
  queried for the PR head commit.
* A required-check summary derived from the actual CI job list. The default
  patterns cover the main test matrix, workflow automation probes, style gate,
  and the `gate / all-required-green` aggregator.
* A job-by-job table covering the latest CI and Docker runs, complete with
  badge emojis indicating state and deep links to the workflow/job logs. Failing
  rows are bolded for quick scanning.
* Coverage headline metrics – latest averages, worst-job coverage, and deltas
  vs the previous recorded run when history is available – along with any
  Markdown snippet published as the `coverage-summary` artifact.

The helper stores the rendered Markdown as
`summary_artifacts/comment_preview.md` so maintainers can inspect the message
directly from the Actions UI. Re-runs overwrite the same preview file and update
the existing PR comment in place.

### Data sources

The workflow collects status data from three places:

1. **Workflow run metadata** – `actions/github-script` queries the REST API to
   locate the most recent CI and Docker runs for the PR head SHA, then expands
   each run's job list so the summary reflects the real workflow structure.
2. **Coverage summary artifact** – if the CI workflow uploads a
   `coverage-summary` artifact, its Markdown payload is embedded verbatim below
   the coverage headline metrics.
3. **Coverage trend records** – JSON and NDJSON artifacts
   (`coverage-trend.json` and `coverage-trend-history.ndjson`) are used to derive
   the latest and previous coverage values, enabling the delta calculations in
   the summary.

### Idempotency & Anti-Spam

* The workflow uses a concurrency group keyed by the head SHA to cancel stale
  runs.
* Comment discovery matches the `### Automated Status Summary` heading, so
  reruns patch the original comment instead of posting duplicates.
* If neither CI nor Docker has produced artifacts yet, the helper still posts a
  pending table so reviewers can see progress.

### Adjusting Behaviour

* Update required-check labelling or job patterns by editing the
  `DEFAULT_REQUIRED_JOB_GROUPS` declaration inside `tools/post_ci_summary.py`,
  or by supplying a `REQUIRED_JOB_GROUPS_JSON` workflow environment override.
* Additional artifacts can be surfaced by extending the artifact download steps
  in `maint-30-post-ci-summary.yml` and updating the Markdown rendering helpers
  inside `tools/post_ci_summary.py`.
* To force Docker to run on documentation-only changes, tweak the `paths-ignore`
  list in `.github/workflows/pr-12-docker-smoke.yml` (the skip rules are
  unchanged by this consolidation).

### Disabling the Summary

Delete or rename `.github/workflows/maint-30-post-ci-summary.yml` to disable the
follower. The consolidated helper is only invoked from that workflow, so no
other automation will recreate the comment once the file is removed.