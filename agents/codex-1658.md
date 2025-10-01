# Issue #1658 – Post-CI Summary Consolidation Log

## Task Checklist

- [x] Inventory legacy summarizer workflows and document their responsibilities.
- [x] Design the unified follower (`maint-30-post-ci-summary.yml`) to listen for
  `workflow_run` events from both CI and Docker.
- [x] Implement artifact download, Markdown rendering, and comment upsert logic
  in `tools/post_ci_summary.py`.
- [x] Fetch and parse coverage and failure snapshot artifacts to feed the new
  summary comment.
- [x] Remove the legacy `maint-31-pr-status-summary.yml` /
  `maint-32-ci-matrix-summary.yml` workflows in favour of the consolidated file.
- [x] Add regression tests for the comment-building helpers and artifact
  parsers.
- [x] Update documentation (`WORKFLOW_GUIDE.md`, `docs/ops/ci-status-summary.md`,
  `docs/ci-failure-tracker.md`, `WORKFLOW_AUDIT_TEMP.md`) to reflect the new
  workflow topology.

## Workflow Inventory

| Workflow | Replaced by | Notes |
|----------|-------------|-------|
| `maint-31-pr-status-summary.yml` | `maint-30-post-ci-summary.yml` | Formerly posted required vs optional job summary. Responsibilities folded into the consolidated script. |
| `maint-32-ci-matrix-summary.yml` | `maint-30-post-ci-summary.yml` | Previously generated Markdown/JSON artifacts for coverage + failure snapshots. Superseded by the single PR comment. |

## Unified Workflow Highlights

- Triggered when CI or Docker finishes (`workflow_run` events) against pull
  requests.
- Downloads shared artifacts into `summary_artifacts/` for coverage trend,
  coverage history, and failure snapshots.
- Calls `tools/post_ci_summary.py` to render the comment, persist a preview
  artifact, and upsert the `### Automated Status Summary` block on the PR.
- Includes regression tests (`tests/test_post_ci_summary.py`) that lock in the
  job table ordering, coverage formatting, and artifact parsing routines.
