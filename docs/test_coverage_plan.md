# Test Coverage Improvement Plan

## Scope and Key Constraints
- Raise overall project test coverage from the current baseline (48.64% per CI run 18016554334) to at least 95%.
- Ensure no tracked module remains with what the team classifies as "low" coverage (define as <90%) once improvements land.
- Focus on production Python packages under `src/` and the public-facing CLI utilities; exclude demo notebooks and archived workflows unless required for coverage uplift.
- Maintain existing behaviour and public APIs — only add or adjust tests or supporting test fixtures/utilities unless defects are uncovered.
- Keep the suite runnable via existing tooling (`pytest`, `scripts/run_tests.sh`) without introducing new required dependencies; optional dev-only helpers must be added to extras.
- Document any uncovered defects found while adding tests and coordinate fixes per repository workflow (e.g., `phase2-dev` for core ranking changes).

## Acceptance Criteria / Definition of Done
- CI coverage soft gate reports >=95% aggregate coverage and no file flagged below the 90% minimum threshold.
- All newly added or modified tests pass locally (`pytest`) and in CI.
- Any identified bugs exposed by new tests are either fixed with accompanying tests or logged with blocking issues referencing the failing scenarios.
- Test documentation updated (e.g., `TESTING_SUMMARY.md`) if new test suites, fixtures, or execution steps are introduced.
- No regressions in existing functionality as validated by the standard test suite and smoke commands defined in project docs.

## Initial Task Checklist
1. **Baseline Audit**
   - [ ] Pull the latest coverage artifacts (HTML/JSON) from CI run 18016554334 to pinpoint modules <90%.
   - [ ] Inventory existing tests by package to map gaps (CLI paths, data pipelines, config validation, etc.).

2. **Prioritised Test Development**
   - [ ] Draft targeted test cases for the lowest-coverage modules first (e.g., any files currently at 0%).
   - [ ] Add fixtures/mocks for external integrations (data IO, CLI entrypoints) to enable deterministic coverage.
   - [ ] Validate edge cases uncovered during planning (error handling, boundary inputs, config permutations).

3. **Iterative Execution and Tracking**
   - [ ] Run `pytest` with coverage locally to confirm incremental gains; adjust thresholds as needed.
   - [ ] Update coverage tracking spreadsheet or issue checklist to reflect improvements per module.
   - [ ] Re-run CI (via existing workflows) to verify the aggregate coverage and file-level minimums.

4. **Documentation and Handover**
   - [ ] Summarise coverage deltas, remaining gaps, and follow-up actions in `TESTING_SUMMARY.md` or the associated issue (#1630).
   - [ ] Capture lessons learned / reusable fixtures for future contributors.
