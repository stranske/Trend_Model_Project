# Workflow Naming Guide (WFv1)

This repository adopts the **Workflow File naming v1 (WFv1)** convention to
keep GitHub Actions organized and predictable. Each workflow file name begins
with a category prefix and a two-digit index so related flows sort together in
GitHub’s UI and in git diffs.

## Category Prefixes

| Prefix | Scope | Typical Triggers | Examples |
|--------|-------|------------------|----------|
| `pr-1x-*` | Pull request checks and fast feedback | `pull_request`, `push` to default branches | `pr-10-ci-python.yml`, `pr-12-docker-smoke.yml` |
| `maint-3x-*` | Repository maintenance, hygiene, and reporting | `schedule`, `workflow_run`, governance automations | `maint-30-post-ci-summary.yml`, `maint-33-check-failure-tracker.yml`, `maint-35-repo-health-self-check.yml`, `maint-40-ci-signature-guard.yml` |
| `agents-4x-*` | Issue and agent orchestration workflows | `issues`, `pull_request_target`, manual diagnostics | `agents-40-consumer.yml`, `agents-41-assign-and-watch.yml` (+ wrappers `agents-41-assign.yml` / `agents-42-watchdog.yml`) |
| `reusable-9x-*` | Reusable building blocks invoked via `workflow_call` | `workflow_call`, `workflow_dispatch` | `reusable-90-agents.yml`, `reusable-ci-python.yml`, `reusable-99-selftest.yml` |

> **Numbering tips**
> * Use the first digit (`1`, `3`, `4`, `9`) to select the category.
> * The second digit groups related flows (e.g. `pr-10`, `pr-12` for the
>   primary PR lane).
> * Leave gaps when possible so future additions can slot in without renumbering.
> * `reusable-ci-python.yml` keeps its legacy slug temporarily for consumer
>   compatibility; future reusable additions should follow the `reusable-9x-*`
>   format directly.

## Stable Workflow Names

Renaming files does **not** change the user-visible workflow names shown in the
Actions tab. We intentionally keep important names such as **CI** and
**Docker** unchanged so existing `workflow_run` triggers (for status summaries
and failure tracking) continue to function without updates.

## Adding or Renaming Workflows

1. Pick the category prefix from the table above and choose the next available
   two-digit slot.
2. Update references in documentation, scripts, and tests to the new file name.
3. Keep workflow `name:` values stable unless there is a deliberate reason to
   change downstream triggers.
4. If you restructure triggers (e.g. `workflow_run` followers), audit the
   `workflows:` lists to make sure they continue to target the **workflow
   names** ("CI", "Docker", etc.).

For deeper operational context, see [`.github/workflows/README.md`](.github/workflows/README.md)
for topology diagrams and orchestration details.

## maint-30-post-ci-summary.yml — Post CI Summary

**Trigger:** `workflow_run` on the `CI` and `Docker` workflows after completion. The
workflow only acts on pull-request runs and de-duplicates executions by head
SHA via concurrency control.

**Responsibilities:**

- Collect job status metadata for the latest CI and Docker workflow runs that
  share the current head SHA.
- Download shared artifacts from both runs (coverage trend + summary, failure
  snapshot, etc.) so the resulting comment mirrors the legacy PR status and CI
  matrix summaries.
- Render a single "Automated Status Summary" Markdown block that merges the
  job table, required-check rollup, coverage deltas, and open failure issue
  details.
- Upsert (create or update) the canonical pull-request comment and expose the
  rendered body as a job summary preview for quick inspection in Actions logs.

**Migration notes:**

- Retires `maint-31-pr-status-summary.yml` and `maint-32-ci-matrix-summary.yml`
  in favour of a single post-CI summarizer.
- Comment identity now keys off the `### Automated Status Summary` heading
  rather than an HTML marker; the helper will upsert the same comment on every
  rerun.
- The `summary_artifacts/` directory retains a copy of the generated Markdown
  (`comment_preview.md`) so other diagnostics can reuse the message without
  hitting the GitHub API.
- Regression coverage for comment formatting and artifact parsing lives in
  `tests/test_post_ci_summary.py`; extend these tests when adjusting table
  formats, coverage calculations, or artifact names.
