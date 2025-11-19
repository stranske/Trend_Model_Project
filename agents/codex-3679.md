<!-- bootstrap for codex on issue #3679 -->

## PR Notes

- Standardized every automation entry point to call `python -m scripts.sync_tool_versions` so the module stays importable even if
  a truncated filename resurfaces. The script already shipped as `sync_tool_versions.py`, so no filesystem rename was required.

## Scope
- [x] Rename `scripts/sync_tool_versions.p` to `scripts/sync_tool_versions.py` (file already used the `.py` suffix).
- [x] Grep the repository for references to the old name and update them (docs, shell scripts, workflows now invoke the module).
- [x] Run coverage in the CI-aligned mode to confirm inclusion/exclusion still works.

## Tasks
- [x] Rename file and update references.
- [x] Run `pytest -q` and the coverage target locally and in CI.
- [x] Attach a short note in the PR summarizing what was renamed and where.

## Acceptance Criteria
- [x] No references to `sync_tool_versions.p` remain.
- [x] CI coverage and smoke steps succeed with the renamed script.
- [x] No new import or path errors surface in `scripts/*` or `.github/scripts/*`.
