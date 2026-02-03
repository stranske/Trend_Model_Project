# Keepalive Status — PR #4672

## Scope
Not provided.

## Checklist Reconciliation
Checklist reconciled on 2026-02-03 (mc list empty-registry output covered).

## Tasks
- [x] Add `trend mc` command group and subcommands in `src/trend_analysis/cli.py`.
- [x] Define scope for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Define scope for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Define scope for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement `mc list` with tag filtering and table/json output.
- [x] Implement `mc validate` to load scenarios and surface validation errors.
- [x] Implement `mc run` with overrides, dry-run mode, progress, and manifest writing.
- [x] Add CLI tests for exit codes, overrides, and output bundle creation.
- [x] Add CLI tests for exit codes. (verify: tests pass)
- [x] Add CLI tests for parameter overrides. (verify: tests pass)
- [x] Add CLI tests for output bundle creation. (verify: tests pass)
- [x] ## Output
- [x] ### List Output (Table)
- [x] ### Run Progress
- [x] Add `trend mc` command group and subcommands in `src/trend_analysis/cli.py`.
- [x] Define scope for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc list` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Define scope for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc validate` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Define scope for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Add `trend mc run` subcommand in `src/trend_analysis/cli.py`. (verify: confirm completion in repo)
- [x] Implement `mc list` with tag filtering and table/json output.
- [x] Implement `mc validate` to load scenarios and surface validation errors.
- [x] Implement `mc run` with overrides, dry-run mode, progress, and manifest writing.
- [x] Add CLI tests for exit codes, overrides, and output bundle creation.
- [x] Add CLI tests for exit codes. (verify: tests pass)
- [x] Add CLI tests for parameter overrides. (verify: tests pass)
- [x] Add CLI tests for output bundle creation. (verify: tests pass)

## Acceptance Criteria
- [x] `trend mc list` shows all registered scenarios.
- [x] `trend mc list --tags X` filters correctly.
- [x] `trend mc validate` reports errors clearly.
- [x] `trend mc run` executes scenario and writes bundle.
- [x] Parameter overrides work (n_paths, jobs, seed).
- [x] Dry run validates without executing.
- [x] Progress bar shows during execution.
- [x] Run manifest written with parameters used.
- [x] Exit codes: 0 = success, 1 = validation error, 2 = runtime error.
- [x] `trend mc list` shows all registered scenarios
- [x] `trend mc list --tags X` filters correctly
- [x] `trend mc validate` reports errors clearly
- [x] `trend mc run` executes scenario and writes bundle
- [x] Parameter overrides work (n_paths, jobs, seed)
- [x] Dry run validates without executing
- [x] Progress bar shows during execution
- [x] Run manifest written with parameters used
- [x] Exit codes: 0 = success, 1 = validation error, 2 = runtime error
- [x] ## Files to Create/Modify
- [x] Extend `src/trend_analysis/cli.py`
- [x] `tests/test_cli_mc.py`
