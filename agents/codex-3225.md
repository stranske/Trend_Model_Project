<!-- bootstrap for codex on issue #3225 -->

# Tool Pin Maintenance Plan (Issue #3225)

## Scope & Key Constraints
- Refresh the CI autofix/tooling pins to the latest released versions highlighted by the Tool Version Check (currently scoped to formatters, type checking, and test runners in `.github/workflows/autofix-versions.env`).
- Keep the pin file, `pyproject.toml`, and `requirements.txt` in lock-step by relying on the existing `scripts/sync_tool_versions.py` helper.
- Avoid unrelated dependency bumps or workflow edits; only update packages that actually have newer upstream releases.
- Validate the updated pins locally where feasible without introducing new tooling beyond the standard scripts in this repo.

## Acceptance Criteria / Definition of Done
- [x] `.github/workflows/autofix-versions.env` records the latest available versions for each out-of-date tool identified by Issue #3225 (e.g., `ruff==0.14.3`).
- [x] Running `python scripts/sync_tool_versions.py --check` succeeds, confirming that `pyproject.toml` and `requirements.txt` mirror the pin file.
- [x] The plan below is updated to mark completed work, including final task status reflecting the delivered pin updates.

## Task Checklist & Status
- [x] Verify latest PyPI releases for the pinned tools (black, ruff, isort, docformatter, mypy, pytest, pytest-cov, coverage).
- [x] Update `.github/workflows/autofix-versions.env` with any new versions (expected: bump `RUFF_VERSION` to 0.14.3).
- [x] Run `python scripts/sync_tool_versions.py --apply` to propagate the pin changes to `pyproject.toml` and `requirements.txt`.
- [x] Run `python scripts/sync_tool_versions.py --check` to ensure all tracked files are in sync.
- [x] Capture verification notes in the Progress Log and mark completed tasks/acceptance criteria.

## Progress Log
- **2025-11-03** – Queried PyPI via `python -m pip index versions <package>` for each pinned tool; only Ruff required an update (0.14.2 → 0.14.3).
- **2025-11-03** – Bumped `RUFF_VERSION` in `.github/workflows/autofix-versions.env`, ran `python scripts/sync_tool_versions.py --apply`, and validated alignment with `--check` (exit status 0).
