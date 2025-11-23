# Keepalive Status — PR #3758

> **Status:** Risk-free configuration implemented and covered by tests.

## Scope
- [x] Add a configuration option to specify the risk-free column name, with validation to ensure it exists and is excluded from investable assets.
- [x] Make the current lowest-volatility fallback opt-in and clearly logged when used.
- [x] Update documentation and example configs to demonstrate explicit risk-free selection.

## Tasks
- [x] Add config parsing/validation for an explicit risk-free column and gate the heuristic behind a flag.
- [x] Adjust risk calculations to use the configured series and exclude it from asset universes.
- [x] Update docs/sample configs and add tests covering explicit selection and the optional fallback.

## Acceptance criteria
- [x] Runs fail fast with a clear error if the configured risk-free column is missing.
- [x] Risk metrics and scaling use the configured series, with tests covering both explicit and opt-in heuristic modes.
