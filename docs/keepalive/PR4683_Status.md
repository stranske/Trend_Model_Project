# Keepalive Status — PR #4683

## Scope
Support multiple folds (vintages) in Monte Carlo scenarios to test robustness across different calibration/forecast periods.

## Progress
43/43 tasks complete, 0 remaining.

## Checklist Reconciliation
Checklist reconciled on 2026-02-03 (recent commit was formatting-only in tests; no task status changes).

## Tasks
- [x] Implement `Fold` class in `src/trend_analysis/monte_carlo/folds.py`.
- [x] Implement `FoldGenerator` class with explicit mode in `src/trend_analysis/monte_carlo/folds.py`.
- [x] Extend `FoldGenerator` to support rolling mode in `src/trend_analysis/monte_carlo/folds.py`.
- [x] Extend `FoldGenerator` to support count_spaced mode in `src/trend_analysis/monte_carlo/folds.py`.
- [x] Integrate fold generation into `runner.py` with correct calibration windows.
- [x] Modify `export.py` to include fold IDs in result tables.
- [x] Define scope for: Identify locations in `export.py` where fold IDs need to be added. (verify: confirm completion in repo)
- [x] Implement focused slice for: Identify locations in `export.py` where fold IDs need to be added. (verify: confirm completion in repo)
- [x] Validate focused slice for: Identify locations in `export.py` where fold IDs need to be added. (verify: confirm completion in repo)
- [x] Define scope for: Implement logic to include fold IDs in result tables. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement logic to include fold IDs in result tables. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement logic to include fold IDs in result tables. (verify: confirm completion in repo)
- [x] Define scope for: Test the integration of fold IDs in result tables. (verify: confirm completion in repo)
- [x] Implement focused slice for: Test the integration of fold IDs in result tables. (verify: confirm completion in repo)
- [x] Validate focused slice for: Test the integration of fold IDs in result tables. (verify: confirm completion in repo)
- [x] Modify `export.py` to add cross-fold summaries.
- [x] Define the structure (verify: confirm completion in repo)
- [x] content of cross-fold summaries. (verify: confirm completion in repo)
- [x] Define scope for: Implement logic to generate cross-fold summaries in `export.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement logic to generate cross-fold summaries in `export.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement logic to generate cross-fold summaries in `export.py`. (verify: confirm completion in repo)
- [x] Test the generation of cross-fold summaries. (verify: confirm completion in repo)
- [x] Modify `export.py` to add pooled option for distributions.
- [x] Define the behavior (verify: confirm completion in repo)
- [x] configuration for pooled distributions. (verify: config validated)
- [x] Define scope for: Implement logic to support pooled distributions in `export.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement logic to support pooled distributions in `export.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement logic to support pooled distributions in `export.py`. (verify: confirm completion in repo)
- [x] Test the pooled distributions functionality. (verify: confirm completion in repo)
- [x] Add unit tests for explicit mode in `tests/monte_carlo/test_folds.py`.
- [x] Add unit tests for rolling mode in `tests/monte_carlo/test_folds.py`.
- [x] Add unit tests for count_spaced mode in `tests/monte_carlo/test_folds.py`.
- [x] Add unit tests for fold IDs in result tables in `tests/monte_carlo/test_folds.py`.
- [x] Add unit tests for cross-fold summaries in `tests/monte_carlo/test_folds.py`.
- [x] Add unit tests for pooled distributions in `tests/monte_carlo/test_folds.py`.

## Acceptance Criteria
- [x] Fold runs can be enabled/disabled by scenario config.
- [x] All three fold modes work (explicit, rolling, count_spaced).
- [x] Each fold uses correct calibration window for return model fitting.
- [x] Output includes fold ID in all result tables.
- [x] Cross-fold comparison summary is generated.
- [x] Optional pooled distributions are included and clearly labeled.
- [x] Unit tests for fold generation logic pass.
- [x] Unit tests for fold-aware outputs pass.
