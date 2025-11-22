# Keepalive Status for Risk-Free Configuration PR

## Scope
- [ ] Add a configuration option to specify the risk-free column name, with validation to ensure it exists and is excluded from investable assets.
- [ ] Make the current lowest-volatility fallback opt-in and clearly logged when used.
- [ ] Update documentation and example configs to demonstrate explicit risk-free selection.

## Tasks
- [ ] Add config parsing/validation for an explicit risk-free column and gate the heuristic behind a flag.
- [ ] Adjust risk calculations to use the configured series and exclude it from asset universes.
- [ ] Update docs/sample configs and add tests covering explicit selection and the optional fallback.

## Acceptance criteria
- [ ] Runs fail fast with a clear error if the configured risk-free column is missing.
- [ ] Risk metrics and scaling use the configured series, with tests covering both explicit and opt-in heuristic modes.

## Progress notes
- Status unchanged: no scope items, tasks, or acceptance criteria have been met yet. Check items off only after acceptance criteria are satisfied and repost this checklist whenever any box is newly completed.

Status auto-updates as tasks complete on this branch.
