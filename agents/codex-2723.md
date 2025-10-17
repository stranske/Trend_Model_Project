# Codex bootstrap for Issue #2723

[Issue #2723](https://github.com/stranske/Trend_Model_Project/issues/2723)

## Purpose

Provide Codex contributors with a concise overview of the consolidation work that finalizes "Maint 46 Post CI" as the single post-CI summary and coverage aggregator.

## Scope

- Ensure the Maint 46 Post CI workflow triggers only for Gate runs that originate from pull requests.
- Resolve the correct PR head ref and SHA from the Gate payload so downstream steps stay aligned with the latest commit.
- Keep workflow permissions minimal while supporting autofix pushes and comment updates (`contents: write`, `pull-requests: read`, `checks: read`, `issues: write`, `actions: read`). Document any additional scope that becomes necessary.
- Produce one consolidated post-CI summary per PR, including a short coverage section when data is available.

## Tasks

- [ ] Successful run after Gate completes on a PR _(awaiting verification on the next Gate follower run)._ 
- [x] `GITHUB_STEP_SUMMARY` shows concise, useful information.
- [x] No duplicate comments per PR.

## Acceptance Criteria

- A Gate run triggers Maint 46 Post CI which updates one consolidated summary for the PR.
- The summary captures run metadata and coverage details (when present) and handles missing coverage data gracefully.
- Workflow permissions remain scoped to the minimal set required for comment updates, coverage download, and autofix pushes (`contents: write`, `pull-requests: read`, `checks: read`, `issues: write`, `actions: read`).
