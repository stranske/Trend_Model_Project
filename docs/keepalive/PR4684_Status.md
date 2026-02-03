# Keepalive Status — PR #4684

## Scope
PR #4684 addressed issue #4683 but verification identified concerns (verdict: CONCERNS). This follow-up addresses the remaining gaps with improved task structure to ensure comprehensive coverage of fold-aware outputs and robust testing of fold modes.

## Progress
22/37 tasks complete, 15 remaining.

## Checklist Reconciliation
Checklist reconciled on 2026-02-03 after reviewing commit b7222116 and running `pytest tests/monte_carlo/test_results.py -m "not slow"`. Export outputs are implemented in `src/trend_analysis/monte_carlo/results.py` (no standalone `export.py`), and the export points are results, summary, cross-fold summary, and pooled summary frames.
Additional fold-aware output tables live outside Monte Carlo exports in `src/trend_analysis/walk_forward.py` (folds/summary CSVs) and `analysis/cv.py` (cv_folds/cv_summary CSVs). These should include a `fold_id` column while keeping existing `fold`/`folds` fields for compatibility. Validated updates on 2026-02-03 with `pytest tests/test_walk_forward_grid.py tests/test_walk_forward_helpers_additional.py tests/test_walk_forward_settings.py tests/test_cv.py -m "not slow"`.

## Tasks
- [x] Review and update export-related code to ensure fold_id columns are included in all relevant export points, including export.py if necessary.
- [x] Define scope for: Review export.py to identify all relevant export points requiring fold_id columns. (verify: confirm completion in repo)
- [x] Implement focused slice for: Review export.py to identify all relevant export points requiring fold_id columns. (verify: confirm completion in repo)
- [x] Validate focused slice for: Review export.py to identify all relevant export points requiring fold_id columns. (verify: confirm completion in repo)
- [x] Update export.py (verify: confirm completion in repo)
- [x] Define scope for: other relevant files to include fold_id columns. (verify: confirm completion in repo)
- [x] Implement focused slice for: other relevant files to include fold_id columns. (verify: confirm completion in repo)
- [x] Validate focused slice for: other relevant files to include fold_id columns. (verify: confirm completion in repo)
- [x] Define scope for: Verify that fold_id columns are correctly included in all export points through unit tests.
- [x] Implement focused slice for: Verify that fold_id columns are correctly included in all export points through unit tests.
- [x] Validate focused slice for: Verify that fold_id columns are correctly included in all export points through unit tests.
- [x] Add unit tests for rolling and count_spaced fold modes to verify correct computation of fold windows and calibration dates, including edge cases.
- [x] Add unit tests for rolling fold mode to verify fold window
- [x] calibration date computations (verify: confirm completion in repo)
- [x] including edge cases. (verify: confirm completion in repo)
- [x] Add unit tests for count_spaced fold mode to verify fold window
- [x] calibration date computations (verify: confirm completion in repo)
- [x] including edge cases. (verify: confirm completion in repo)
- [x] Enhance unit tests for fold-aware outputs to verify that all exported result tables contain the fold_id column.
- [x] Enhance existing unit tests for fold-aware outputs. (verify: tests pass)
- [x] Define scope for: Add new tests specifically to verify that all exported result tables contain the fold_id column.
- [x] Implement focused slice for: Add new tests specifically to verify that all exported result tables contain the fold_id column.
- [x] Validate focused slice for: Add new tests specifically to verify that all exported result tables contain the fold_id column.
- [ ] Implement additional unit tests for FoldGenerator._align_to_index and FoldGenerator._previous_in_index to cover edge cases.
- [x] Implement unit tests for FoldGenerator._align_to_index (verify: tests pass)
- [ ] including edge cases. (verify: confirm completion in repo)
- [x] Implement unit tests for FoldGenerator._previous_in_index (verify: tests pass)
- [ ] including edge cases. (verify: confirm completion in repo)
- [ ] Clarify and update implementation regarding pooled output to determine if full distribution artifacts are needed, and update tests accordingly.
- [ ] Add or enhance tests to verify that the scenario configuration flag properly enables and disables fold runs.

## Acceptance Criteria
- [ ] The scenario configuration includes a flag that enables or disables fold runs, and when disabled, no fold-related code paths are executed.
- [ ] Explicit, rolling, and count_spaced fold modes are fully functional and covered by unit tests that verify expected fold start, calibration, and window calculations.
- [ ] For each fold, the calibration window is correctly calculated and applied for return model fitting, with unit tests checking both typical and edge-case scenarios.
- [ ] Every result table that is part of fold-aware output includes a fold_id column.
- [ ] A cross-fold comparison summary frame is generated and exported via the designated export interface, with unit tests verifying its presence and correctness.
- [ ] Pooled output clearly indicates whether it represents a distribution of full data or only summary statistics, with outputs labeled with scope 'pooled'.
- [ ] FoldGenerator._align_to_index and FoldGenerator._previous_in_index are covered by unit tests that include edge cases.
