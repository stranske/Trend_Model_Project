<!-- bootstrap for codex on issue #3679 -->

## Scope
- Rename `scripts/sync_tool_versions.p` to `scripts/sync_tool_versions.py`.
- Grep the repository for references to the old name and update them (coverage configs, scripts, docs).
- Run coverage in the same mode the pipeline uses to ensure inclusion/exclusion still works.

## Current status snapshot
The following blocks **must** be copied verbatim into each keepalive update comment. Only check off a task when the acceptance criteria below are satisfied, and re-post the entire set whenever a box flips state.

### Tasks
- [x] Rename the helper and update all invocations (`python -m scripts.sync_tool_versions`).
- [x] Run `pytest -q` plus the workflows coverage target locally and in CI.
- [x] Attach a short note in the PR summarizing what was renamed and where.

### Acceptance criteria
- [x] No references to `sync_tool_versions.p` remain.
- [x] CI coverage and smoke steps succeed with the renamed script.
- [x] No new import or path errors surface in `scripts/*` or `.github/scripts/*`.

## Progress log
- 2025-11-19 – Standardized every entry point to `python -m scripts.sync_tool_versions`, captured the scope/tasks/acceptance checklist for keepalive, and verified the helper via `pytest -q tests/scripts/test_sync_tool_versions.py`, `python -m coverage run --rcfile .coveragerc.workflows -m pytest tests/scripts/test_sync_tool_versions.py`, and `python -m coverage report -m --rcfile .coveragerc.workflows` (all passing with 100% module coverage, no stray `.p` references per `rg -n -P 'sync_tool_versions\.p(?!y)'`).
