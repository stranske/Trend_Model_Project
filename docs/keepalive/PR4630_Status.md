# Keepalive Status — PR #4630

## Scope
PR #4630 addressed issue #4629, but verification identified concerns (verdict: CONCERNS). This follow-up addresses the remaining gaps with improved task structure to meet the acceptance criteria.

## Tasks
- [x] Update `src/trend_analysis/monte_carlo/__init__.py` to explicitly import `MonteCarloScenario` and `MonteCarloSettings` and add them to the `__all__` list.
- [x] Import `MonteCarloScenario` (verify: confirm completion in repo)
- [x] Define scope for: `MonteCarloSettings` in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: `MonteCarloSettings` in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: `MonteCarloSettings` in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Add `MonteCarloScenario` (verify: confirm completion in repo)
- [x] Define scope for: `MonteCarloSettings` to the `__all__` list in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Implement focused slice for: `MonteCarloSettings` to the `__all__` list in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Validate focused slice for: `MonteCarloSettings` to the `__all__` list in `src/trend_analysis/monte_carlo/__init__.py`. (verify: confirm completion in repo)
- [x] Modify the `load_scenario` function in the MonteCarlo scenario parser to include null-checks for the 'return_model' and 'folds' parameters, raising a `ValueError` with a descriptive error message if either is `None`.
- [x] Add null-checks for the 'return_model' (verify: confirm completion in repo)
- [x] 'folds' parameters in the `load_scenario` function. (verify: confirm completion in repo)
- [x] Define scope for: Raise a `ValueError` with a descriptive error message if 'return_model' or 'folds' is `None`. (verify: confirm completion in repo)
- [x] Implement focused slice for: Raise a `ValueError` with a descriptive error message if 'return_model' or 'folds' is `None`. (verify: confirm completion in repo)
- [x] Validate focused slice for: Raise a `ValueError` with a descriptive error message if 'return_model' or 'folds' is `None`. (verify: confirm completion in repo)
- [x] Implement or update the `_coerce_tags` function to convert input tags to lowercase, whether the input is a single string or a list of strings. Add or update tests as needed.
- [x] Define scope for: Implement or update the `_coerce_tags` function to convert input tags to lowercase. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement or update the `_coerce_tags` function to convert input tags to lowercase. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement or update the `_coerce_tags` function to convert input tags to lowercase. (verify: confirm completion in repo)
- [x] Define scope for: Add or update tests to verify that `_coerce_tags` converts input tags to lowercase for both single strings
- [x] Implement focused slice for: Add or update tests to verify that `_coerce_tags` converts input tags to lowercase for both single strings
- [x] Validate focused slice for: Add or update tests to verify that `_coerce_tags` converts input tags to lowercase for both single strings
- [x] lists of strings. (verify: confirm completion in repo)

## Acceptance Criteria
- [x] The `src/trend_analysis/monte_carlo/__init__.py` file explicitly imports `MonteCarloScenario` and `MonteCarloSettings` and includes both in the module's `__all__` list.
- [x] The `load_scenario` function raises a `ValueError` with a descriptive message when 'return_model' or 'folds' is `None`.
- [x] The `_coerce_tags` function converts all tags to lowercase, whether provided as a single string or within a list of strings.

## Progress
26/26 tasks complete
