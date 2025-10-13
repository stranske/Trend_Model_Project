# Issue #1658 – Post-CI Summary Consolidation Log

## Task Checklist

### Issue Comment Task List (2026-02-15)

- [x] Inventory the legacy summarizer workflows under `.github/workflows/` and
  capture their responsibilities. → Documented below in the workflow inventory
  table (legacy `maint-31` / `maint-32` versus consolidated `maint-46`).
- [x] Design the unified workflow (`maint-46-post-ci.yml`) so it
  triggers on `workflow_run` events for the "CI" and "Docker" pipelines and
  scopes to pull-request runs. → Implemented in the workflow with concurrency
  guard and head-SHA discovery.
- [x] Implement shared steps that download artifacts, render the consolidated
  Markdown, and upsert the PR comment. → See
  `.github/workflows/maint-46-post-ci.yml` plus
  `tools/post_ci_summary.py` helpers.
- [x] Fetch the required artifacts (coverage trend + summary, failure
  snapshot) so the comment includes coverage deltas and failure listings. →
  `load_coverage_details` / `load_failure_snapshot` consume the artifacts and
  feed the comment builder.
- [x] Render a consolidated Markdown summary combining the historic PR status
  and CI matrix outputs. → `build_comment_body` merges requirement rollups,
  job table, coverage, and failure sections.
- [x] Update (create if missing) the single "Automated Status Summary" comment
  idempotently on reruns. → `upsert_comment` locates the existing marker and
  PATCHes or POSTs accordingly.
- [x] Remove or shim the legacy workflows so duplicate summaries do not run. →
  Legacy files deleted; only the consolidated workflow remains active.
- [x] Add regression coverage for comment formatting and artifact parsing. →
  `tests/test_post_ci_summary.py` exercises the renderers and loaders.
- [x] Document the new workflow (trigger conditions, outputs, migration notes)
  in `WORKFLOW_GUIDE.md` and companion ops docs.

## Workflow Inventory

| Workflow | Replaced by | Notes |
|----------|-------------|-------|
| `maint-31-pr-status-summary.yml` | `maint-46-post-ci.yml` | Formerly posted required vs optional job summary. Responsibilities folded into the consolidated script. |
| `maint-32-ci-matrix-summary.yml` | `maint-46-post-ci.yml` | Previously generated Markdown/JSON artifacts for coverage + failure snapshots. Superseded by the single PR comment. |

## Unified Workflow Highlights

- Triggered when CI or Docker finishes (`workflow_run` events) against pull
  requests.
- Downloads shared artifacts into `summary_artifacts/` for coverage trend,
  coverage history, and failure snapshots.
- Calls `tools/post_ci_summary.py` to render the comment, persist a preview
  artifact, and upsert the `### Automated Status Summary` block on the PR.
- Includes regression tests (`tests/test_post_ci_summary.py`) that lock in the
  job table ordering, coverage formatting, and artifact parsing routines.
