# Keepalive Status — PR #4470

## Scope
Unify CLI entrypoints around `trend`, keep legacy wrappers forwarding with deprecation warnings, and document the migration.

## Tasks
- [x] Identify redundant scripts in `pyproject.toml` `[project.scripts]` and map each to a `trend <subcommand>` equivalent.
- [x] Add wrapper functions in `src/trend/compat_entrypoints.py` that call `trend.cli:main([...])` with preset argv and emit a deprecation warning.
- [x] Update `pyproject.toml` to point legacy scripts at wrappers where applicable.
- [x] Update `README.md` and `docs/ReproducibilityGuide.md` to recommend `trend` and include a migration table (old -> new).
- [x] Add a minimal smoke test that wrapper entrypoints return `--help` successfully (or at least import and run without error).

## Acceptance Criteria
- [x] `trend` covers core workflows (run/report/app/quick-report).
- [x] Legacy scripts still function and forward to `trend` with a clear warning.
- [x] Docs prefer `trend`, and tests pass.
