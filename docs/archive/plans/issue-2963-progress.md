# Issue #2963 Progress

## Scope
- [x] For each inline script longer than ~15 lines, move the logic into a dedicated helper file under `.github/scripts/`.
- [x] Add minimal tests that validate JSON I/O and guard conditions for the extracted helpers.

## Tasks
- [x] Create `.github/scripts/` with documentation describing usage.
- [x] Extract helper scripts for Gate summary aggregation, branch-protection snapshot restoration, and Agents glue code.
- [x] Add a lightweight pytest job that executes the Node and Python unit tests for the helper scripts.

## Acceptance criteria
- [x] All multi-dozen-line `actions/github-script` blocks have been replaced by helper modules.
- [x] Unit tests for the extracted helper scripts pass locally and in CI.
- [x] Helper behavior matches historical Gate and Agents logs.
