## Automated Post-CI Status Summary

The repository publishes a consolidated status block to the run summary whenever
the `maint-30-post-ci-summary.yml` follower completes. The workflow subscribes
to `workflow_run` events for the `CI` and `Docker` workflows, downloads shared
artifacts, and renders Markdown headed by `## Automated Status Summary` before
appending it to `$GITHUB_STEP_SUMMARY`.

### Summary Contents

Each refresh includes:

* Head commit SHA and a roll-up of the most recent workflow runs that were
  queried for the PR head commit.
* A required-check summary derived from the actual CI job list. The default
  pattern now targets the unified `ci / python` job to match the consolidated
  workflow structure.
* A job-by-job table covering the latest CI and Docker runs, complete with
  badge emojis indicating state and deep links to the workflow/job logs. Failing
  rows are bolded for quick scanning.
* Coverage headline metrics – latest averages, worst-job coverage, and deltas
  vs the previous recorded run when history is available – along with any
  Markdown snippet published as the `coverage-summary` artifact.

The helper stores the rendered Markdown as
`summary_artifacts/summary_preview.md` so maintainers can inspect the message
directly from the Actions UI. Re-runs overwrite the same preview file with the
latest content.

### Data sources

The workflow collects status data from three places:

1. **Workflow run metadata** – `actions/github-script` queries the REST API to
   locate the most recent CI and Docker runs for the PR head SHA, then expands
   each run's job list so the summary reflects the real workflow structure.
2. **Coverage summary artifact** – the CI workflow now writes coverage metrics
   directly to the job summary, but any uploaded Markdown snippet is still
   embedded below the headline metrics when present.
3. **Coverage trend records** – JSON and NDJSON artifacts
   (`coverage-trend.json` and `coverage-trend-history.ndjson`) continue to drive
   the latest and previous coverage values when available.

Missing artifacts now fail soft: the workflow marks the related sections as
pending and logs a notice instead of erroring out when coverage or Docker
bundles have not been published yet.

### Idempotency & Anti-Spam

* The workflow uses a concurrency group keyed by the PR number when Actions
  provides it, falling back to the head SHA and ultimately the
  `workflow_run` identifier, so stale runs are cancelled without clobbering
  other PRs even when GitHub omits the commit hash.
* Because the summary is appended to `$GITHUB_STEP_SUMMARY`, reruns simply
  overwrite the section within the same workflow execution instead of creating
  duplicate PR noise.
* If neither CI nor Docker has produced artifacts yet, the helper still writes a
  pending table so reviewers can see progress.

### Adjusting Behaviour

* Update required-check labelling or job patterns by editing the
  `DEFAULT_REQUIRED_JOB_GROUPS` declaration inside `tools/post_ci_summary.py`,
  or by supplying a `REQUIRED_JOB_GROUPS_JSON` workflow environment override.
* Additional artifacts can be surfaced by extending the artifact download steps
  in `maint-30-post-ci-summary.yml` and updating the Markdown rendering helpers
  inside `tools/post_ci_summary.py`.
* To force Docker to run on documentation-only changes, tweak the `paths-ignore`
  list in `.github/workflows/pr-gate.yml` (the skip rules are unchanged by this
  consolidation).

### Disabling the Summary

Delete or rename `.github/workflows/maint-30-post-ci-summary.yml` to disable the
follower. The consolidated helper is only invoked from that workflow, so no
other automation will recreate the run-summary entry once the file is removed.
