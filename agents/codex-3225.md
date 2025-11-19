<!-- bootstrap for codex on issue #3225 -->

# Codex Agent Bootstrap for Issue #3225

This document bootstraps the Codex agent workstream scoped to
[issue #3225](https://github.com/Trend_Model_Project/Trend_Model_Project/issues/3225).
It explains the agent's purpose, operating assumptions, and the concrete
delivery plan for the associated tooling maintenance effort.

## Purpose

The Codex agent coordinates the recurring maintenance of our shared toolchain
pins (formatters, linters, and test runners) so local development and CI remain
in sync. Updates are limited to packages monitored by the Tool Version Check in
`.github/workflows/autofix-versions.env`.

## Usage Instructions

- Ensure the base project environment is provisioned following
  [`scripts/setup_env.sh`](../scripts/setup_env.sh).
- Use `python -m scripts.sync_tool_versions --apply` to propagate version bumps
  from the pin file into `pyproject.toml` and `requirements.lock`.
- Validate the synchronization with `python -m scripts.sync_tool_versions --check`
  before marking the tasks below complete.

## Configuration Details

- No bespoke configuration is required for this bootstrap; it relies on the
  repository's standard scripts and configuration files.
- Tool-specific settings continue to live alongside the main project
  configuration (e.g., `pyproject.toml` for Ruff and Black settings).

# Tool Pin Maintenance Plan (Issue #3225)

## Scope & Key Constraints
- Refresh the CI autofix/tooling pins to the latest released versions highlighted by the Tool Version Check (currently scoped to formatters, type checking, and test runners in `.github/workflows/autofix-versions.env`).
- Keep the pin file, `pyproject.toml`, and `requirements.lock` in lock-step by relying on the existing `scripts/sync_tool_versions.py` helper.
- Avoid unrelated dependency bumps or workflow edits; only update packages that actually have newer upstream releases.
- Validate the updated pins locally where feasible without introducing new tooling beyond the standard scripts in this repo.

## Acceptance Criteria / Definition of Done
- [x] `.github/workflows/autofix-versions.env` records the latest available versions for each out-of-date tool identified by Issue #3225 (e.g., `ruff==0.14.3`).
- [x] Running `python -m scripts.sync_tool_versions --check` succeeds, confirming that `pyproject.toml` and `requirements.lock` mirror the pin file.
- [x] The plan below is updated to mark completed work, including final task status reflecting the delivered pin updates.

## Task Checklist & Status
- [x] Verify latest PyPI releases for the pinned tools (black, ruff, isort, docformatter, mypy, pytest, pytest-cov, coverage).
- [x] Update `.github/workflows/autofix-versions.env` with any new versions (expected: bump `RUFF_VERSION` to 0.14.3).
- [x] Run `python -m scripts.sync_tool_versions --apply` to propagate the pin changes to `pyproject.toml` and `requirements.lock`.
- [x] Run `python -m scripts.sync_tool_versions --check` to ensure all tracked files are in sync.
- [x] Capture verification notes in the Progress Log and mark completed tasks/acceptance criteria.

## Progress Log
- **2025-11-03** – Queried PyPI via `python -m pip index versions <package>` for each pinned tool; only Ruff required an update (0.14.2 → 0.14.3).
- **2025-11-03** – Bumped `RUFF_VERSION` in `.github/workflows/autofix-versions.env`, ran `python -m scripts.sync_tool_versions --apply`, and validated alignment with `--check` (exit status 0).
