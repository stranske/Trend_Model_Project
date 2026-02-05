# Keepalive Status — PR #4721

## Scope
Implement the aggregator that computes distribution summaries over simulated outcomes.

## Checklist Reconciliation
Checklist reconciled on 2026-02-05 after adding percent-string and empty-string quantile coverage and verifying per-strategy path exports with `pytest tests/monte_carlo/test_aggregator.py -m "not slow"`. Re-verified on 2026-02-05 after adding summary-quantile column coverage. Re-verified on 2026-02-05 after adding all-non-finite expected shortfall coverage. Re-verified on 2026-02-05 after adding all-non-finite quantile coverage for per-metric path counts. Re-verified on 2026-02-05 after excluding pooled summary columns (`paths`, `folds`) from metric aggregation and running `pytest tests/monte_carlo/test_aggregator.py -m "not slow"`. Re-verified on 2026-02-05 after adding all-non-finite quantile coverage for all metrics and running `pytest tests/monte_carlo/test_aggregator.py -m "not slow"`. Re-verified on 2026-02-05 after adding fractional quantile summary coverage and running `pytest tests/monte_carlo/test_aggregator.py -m "not slow"`.

## Tasks
- [x] Define aggregation result schema for per-path, quantiles, breach, and expected shortfall tables in `src/trend_analysis/monte_carlo/aggregator.py`.
- [x] Define schema for per-path aggregation results. (verify: confirm completion in repo)
- [x] Define schema for quantiles aggregation results. (verify: confirm completion in repo)
- [x] Define schema for breach probabilities aggregation results. (verify: confirm completion in repo)
- [x] Define schema for expected shortfall aggregation results. (verify: confirm completion in repo)
- [x] Implement aggregator computations for quantiles, breach probabilities, and expected shortfall (ES) on existing metrics in `src/trend_analysis/monte_carlo/aggregator.py`.
- [x] Implement computation for quantiles on existing metrics. (verify: confirm completion in repo)
- [x] Define scope for: Implement computation for breach probabilities on existing metrics. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement computation for breach probabilities on existing metrics. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement computation for breach probabilities on existing metrics. (verify: confirm completion in repo)
- [x] Define scope for: Implement computation for expected shortfall (ES) on existing metrics. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement computation for expected shortfall (ES) on existing metrics. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement computation for expected shortfall (ES) on existing metrics. (verify: confirm completion in repo)
- [x] Add CSV/Parquet export functionality for all aggregation outputs in `src/trend_analysis/monte_carlo/export.py`.
- [x] Add CSV export functionality for aggregation outputs. (verify: confirm completion in repo)
- [x] Add Parquet export functionality for aggregation outputs. (verify: confirm completion in repo)
- [x] Write unit tests for aggregation correctness (quantiles, breach probabilities, ES) in `tests/monte_carlo/test_aggregator.py`.
- [x] Define aggregation result schema for per-path, quantiles, breach, and expected shortfall tables.
- [x] Implement aggregator computations for quantiles, breach probabilities, and ES on existing metrics.
- [x] Add CSV/Parquet export for all aggregation outputs.
- [x] Add unit tests for aggregation correctness (quantiles, breach, ES).

## Acceptance Criteria
- [x] Per-strategy-path table written with columns: strategy, path, fold, metrics.
- [x] Summary quantiles table written with configurable quantiles.
- [x] Breach probability table written for configured thresholds.
- [x] Expected shortfall computed for tail metrics.
- [x] All outputs available in both parquet and CSV formats.
- [x] Metric definitions consistent with existing pipeline metrics.
- [x] Unit tests pass for aggregation correctness.
- [x] Per-strategy-path table written (strategy, path, fold, metrics)
- [x] Summary quantiles table written (configurable quantiles)
- [x] Breach probability table written for configured thresholds
- [x] Expected shortfall computed for tail metrics
- [x] All outputs in both parquet and CSV formats
- [x] Metric definitions consistent with existing pipeline metrics
- [x] Unit tests for aggregation correctness
- [x] ## Files to Create/Modify
- [x] `src/trend_analysis/monte_carlo/aggregator.py`
- [x] `src/trend_analysis/monte_carlo/export.py`
- [x] `tests/monte_carlo/test_aggregator.py`

## Progress
39/39 tasks complete (re-verified 2026-02-05 via `pytest tests/monte_carlo/test_aggregator.py -m "not slow"`; added summary-quantile column coverage).
