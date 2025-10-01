## Automated Post-CI Status Summary

The repository maintains a single, continuously updated pull-request comment
that surfaces CI and Docker results once both workflows finish. The
`maint-30-post-ci-summary.yml` follower reacts to `workflow_run` events for the
`CI` and `Docker` workflows, downloads shared artifacts, and renders a unified
Markdown block headed by `### Automated Status Summary`.

### Comment Contents

Each refresh includes:

* Head commit SHA and the workflow name that triggered the refresh.
* A rollup of required checks (currently the CI test suite, workflow automation
  probes, style gate, and the `gate / all-required-green` aggregator).
* A job-by-job table covering the latest CI and Docker runs, with log links and
  failure rows emphasised.
* Coverage headline metrics – latest averages, worst-job coverage, and deltas
  vs the previous recorded run when history is available – plus the raw table
  emitted by the coverage uploader.
* An optional failure-signature table populated from `ci_failures_snapshot.json`
  when the failure tracker reports open issues.

The helper stores the rendered Markdown as
`summary_artifacts/comment_preview.md` so maintainers can inspect the message
directly from the Actions UI. Re-runs overwrite the same preview file and update
the existing PR comment in place.

### Idempotency & Anti-Spam

* The workflow uses a concurrency group keyed by the head SHA to cancel stale
  runs.
* Comment discovery matches the `### Automated Status Summary` heading, so
  reruns patch the original comment instead of posting duplicates.
* If neither CI nor Docker has produced artifacts yet, the helper still posts a
  pending table so reviewers can see progress.

### Adjusting Behaviour

* Update required-check labelling or job patterns by editing the
  `WORKFLOW_CONFIGS` declaration inside `tools/post_ci_summary.py`.
* Additional artifacts can be surfaced by teaching
  `load_coverage_details` / `load_failure_snapshot` (or adding new helpers) to
  parse them and extending the Markdown rendering functions.
* To force Docker to run on documentation-only changes, tweak the `paths-ignore`
  list in `.github/workflows/pr-12-docker-smoke.yml` (the skip rules are
  unchanged by this consolidation).

### Disabling the Summary

Delete or rename `.github/workflows/maint-30-post-ci-summary.yml` to disable the
follower. The consolidated helper is only invoked from that workflow, so no
other automation will recreate the comment once the file is removed.