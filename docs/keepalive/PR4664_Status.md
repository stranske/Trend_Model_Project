# Keepalive Status — PR #4664

## Scope
Ensure curated strategy packs can extend freeform weighting parameters without altering global defaults, while keeping variant override validation strict for non-curated strategies. Validate that `StrategyVariant.to_trend_config` preserves weighting configuration when both `portfolio.weighting_scheme` and `portfolio.weighting.name` are supplied.

## Tasks
- [x] Update an example scenario configuration file to reference and include the `hf_equity_curated.yml` file.
- [x] Modify the `variant._deep_merge_overrides` function to conditionally apply the freeform override exception for `('portfolio', 'weighting', 'params')` only when handling curated strategies.
- [x] Define scope for: Update the `variant._deep_merge_overrides` function to support conditional application of the freeform override exception. (verify: confirm completion in repo)
- [x] Implement focused slice for: Update the `variant._deep_merge_overrides` function to support conditional application of the freeform override exception. (verify: confirm completion in repo)
- [x] Validate focused slice for: Update the `variant._deep_merge_overrides` function to support conditional application of the freeform override exception. (verify: confirm completion in repo)
- [x] Implement logic to identify curated strategies (verify: confirm completion in repo)
- [x] Define scope for: apply the freeform override exception only for them. (verify: confirm completion in repo)
- [x] Implement focused slice for: apply the freeform override exception only for them. (verify: confirm completion in repo)
- [x] Validate focused slice for: apply the freeform override exception only for them. (verify: confirm completion in repo)
- [x] Add unit tests that load `hf_equity_curated.yml`, validate each strategy against the schema, and confirm that the global defaults remain unaffected.
- [x] Add unit tests to load `hf_equity_curated.yml` (verify: tests pass)
- [x] validate each strategy against the schema. (verify: confirm completion in repo)
- [x] Define scope for: Add unit tests to confirm that global defaults remain unaffected when `hf_equity_curated.yml` is loaded. (verify: tests pass)
- [x] Implement focused slice for: Add unit tests to confirm that global defaults remain unaffected when `hf_equity_curated.yml` is loaded. (verify: tests pass)
- [x] Validate focused slice for: Add unit tests to confirm that global defaults remain unaffected when `hf_equity_curated.yml` is loaded. (verify: tests pass)
- [x] Implement and add test cases for `StrategyVariant.to_trend_config`. These tests should include scenarios where both `portfolio.weighting_scheme` and `portfolio.weighting.name` are provided.
- [x] Implement the `StrategyVariant.to_trend_config` function. (verify: config validated)
- [x] Define scope for: Add test cases for `StrategyVariant.to_trend_config` to verify correct application of weighting methods.
- [x] Implement focused slice for: Add test cases for `StrategyVariant.to_trend_config` to verify correct application of weighting methods.
- [x] Validate focused slice for: Add test cases for `StrategyVariant.to_trend_config` to verify correct application of weighting methods.

## Acceptance Criteria
- [x] The `hf_equity_curated.yml` file is referenced in the `config/scenarios/monte_carlo/example.yml` file, and executing this scenario loads and applies the curated strategies without modifying global defaults.
- [x] The `variant._deep_merge_overrides` function applies the freeform override exception for `('portfolio', 'weighting', 'params')` only when handling curated strategies.
- [x] Unit tests load `hf_equity_curated.yml`, validate each strategy against the schema, and confirm that global defaults remain unaffected.
- [x] Test cases for `StrategyVariant.to_trend_config` verify the correct application of weighting methods when both `portfolio.weighting_scheme` and `portfolio.weighting.name` are provided, and ensure no unintended global modifications.

## Progress
24/24 tasks complete (verified 2026-02-03 via `pytest tests/monte_carlo/strategy/test_variant.py -m "not slow"`).
