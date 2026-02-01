# Keepalive Status — PR #4641

## Scope
Running N strategies independently on the same simulated path recomputes expensive metrics N times, creating a significant performance bottleneck.

Refactor multi-period evaluation so expensive per-window metric/score computations can be reused across strategies within a path.

## Tasks
- [x] Implement `PathContext` class to cache score frames once per path in `src/trend_analysis/monte_carlo/cache.py`.
- [x] Define the structure (verify: confirm completion in repo)
- [x] attributes of the `PathContext` class. (verify: confirm completion in repo)
- [x] Define scope for: Implement the logic for caching score frames in the `PathContext` class. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement the logic for caching score frames in the `PathContext` class. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement the logic for caching score frames in the `PathContext` class. (verify: confirm completion in repo)
- [x] Create `PathContextCache` class to compute and store required metrics in `src/trend_analysis/monte_carlo/cache.py`.
- [x] Define the structure (verify: confirm completion in repo)
- [x] attributes of the `PathContextCache` class. (verify: confirm completion in repo)
- [x] Implement the logic for computing (verify: confirm completion in repo)
- [x] storing required metrics in the `PathContextCache` class. (verify: confirm completion in repo)
- [x] Update `src/trend_analysis/monte_carlo/runner.py` to consume cached frames for strategy evaluation.
- [x] Implement logic to clear cache after path completion to bound memory usage in `src/trend_analysis/monte_carlo/cache.py`.
- [x] Write unit tests in `tests/monte_carlo/test_cache.py` to validate caching behavior and result equivalence.
- [x] Write unit tests to validate caching behavior. (verify: tests pass)
- [x] Write unit tests to validate result equivalence. (verify: tests pass)
- [x] ## Performance Target
- [x] Running 10 strategies should be **materially faster** than running 10 independent full analyses:
- [x] Target: <2x the time of a single strategy (vs 10x without caching)
- [x] Metric computation is O(1) per path regardless of strategy count
- [x] Implement `PathContext` and cache to compute score frames once per path.
- [x] Update runner to consume cached frames for strategy evaluation.
- [x] Clear cache after path completion to bound memory use.
- [x] Add tests validating caching behavior and result equivalence.

## Acceptance Criteria
- [x] Score frames are computed once per rebalance date per path.
- [x] Strategy evaluation uses cached frames without recomputing metrics.
- [x] Running 10 strategies is <2x the time of running 1 strategy on the same path.
- [x] Numerical results are unchanged compared to baseline full runs.
- [x] Cache is cleared after path completion to ensure bounded memory usage.
- [x] Unit tests confirm caching behavior and correctness.
- [x] Score frames computed once per rebalance date per path
- [x] Strategy evaluation consumes cached frames, not recomputing
- [x] Running 10 strategies is <2x time of 1 strategy on same path
- [x] No change in numerical results vs baseline full runs
- [x] Memory usage bounded (clear cache after path completes)
- [x] Unit tests verify caching behavior and correctness
- [x] ## Files to Create/Modify
- [x] `src/trend_analysis/monte_carlo/cache.py`
- [x] Modify `src/trend_analysis/monte_carlo/runner.py` to use cache
- [x] `tests/monte_carlo/test_cache.py`

## Progress
40/40 tasks complete
