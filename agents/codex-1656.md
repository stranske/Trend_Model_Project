<!-- bootstrap for codex on issue #1656 -->

## Iteration 1 Notes
- Added unified CI style job to `pr-10-ci-python.yml` alongside tests and workflow automation.
- Retired the standalone `pr-11-style-gate.yml` workflow and updated docs/scripts to point to the CI style job.
- Ensured gate aggregation depends on the new style job and adjusted automation tests accordingly.

## Iteration 2 Notes
- Updated `requirements.lock` via `uv pip compile` so the lockfile consistency test passes with the refreshed dependency pins.
- Fixed the local style gate script to call mypy against `src/trend_portfolio_app` (correct package path) using proper indentation.
