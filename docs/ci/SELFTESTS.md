# Selftest: Reusables Reference

Reference: [Issue #2814](https://github.com/stranske/Trend_Model_Project/issues/2814)

`selftest-reusable-ci.yml` is the single nightly and manual surface for
rehearsing the reusable CI matrix. The workflow delegates execution to
`reusable-10-ci-python.yml` with `strategy.fail-fast: false` so each
scenario completes even when individual jobs fail. The aggregate job
publishes a compact markdown table into the workflow summary and uploads
a machine-readable report for deeper inspection.

## Matrix Scenarios

| Scenario | Features Exercised | Expected Artifacts |
| -------- | ------------------ | ------------------ |
| `minimal` | Baseline Ruff/Mypy/pytest sweep | Coverage per Python version |
| `metrics_only` | Metrics export pipeline | Coverage + `ci-metrics` |
| `metrics_history` | Metrics plus history artifact | Coverage + `ci-metrics` + `metrics-history` |
| `classification_only` | Classification tagging | Coverage + `classification` |
| `coverage_delta` | Coverage delta regression guard | Coverage + `coverage-delta` |
| `full_soft_gate` | Full gate (metrics, history, classification, coverage delta, soft gate trend) | Coverage + `ci-metrics` + history/ classification set + coverage trend trio |

Each scenario prefixes artifact names with `sf-<scenario>-…` so the
verification step can detect missing or stray outputs. When dual-runtime
mode is enabled the coverage artifacts repeat for each interpreter
specified via the `python_versions` input.

## Outputs and Summaries

- **Step summary** – The aggregate job appends a markdown table listing
  each scenario, missing artifacts, and unexpected extras. This output is
  the “tiny table” referenced in the acceptance criteria.
- **Self-test report artifact** – Uploaded as `selftest-report`, the
  JSON payload mirrors the summary table and records the overall failure
  count plus GitHub run id for offline review.
- **Comment mode** – When `mode: comment` and `post_to: pr-number` are
  supplied, the workflow posts (or updates) a PR comment marked
  `<!-- selftest-reusable-ci-comment -->` summarising the same table and
  highlighting mismatches.

## Dispatch Notes

- **Triggers** – Nightly cron (`30 6 * * *`) and manual
  `workflow_dispatch`. Manual runs can override headings (`summary_title`,
  `comment_title`), supply a rationale (`reason`), and toggle dual-runtime
  coverage (`mode: dual-runtime`).
- **History opt-in** – Enable the `selftest-report` download via
  `enable_history: true` to store the JSON artifact alongside other run
  outputs for debugging.
- **Failure guards** – Publish job shell steps emit explicit errors when
  the verification table or failure count is missing, when mismatches
  remain, or when the matrix completes with a non-success result. These
  messages surface as log annotations in addition to the comment/summary
  output.
