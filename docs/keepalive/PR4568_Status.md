# Keepalive Status — PR #4568

## Scope
Complete Monte Carlo scenario schema validation, improve error messaging, and ensure scenario examples load with coverage in tests.

## Tasks
- [x] Create a valid example scenario file at `config/scenarios/monte_carlo/example.yml` that adheres to the schema defined by `MonteCarloScenario` and `MonteCarloSettings`.
- [x] Implement complete validation logic in the `MonteCarloScenario` and `MonteCarloSettings` dataclasses, ensuring that missing or invalid fields result in clear, informative error messages.
- [x] Implement validation logic in the `MonteCarloScenario` dataclass. (verify: confirm completion in repo)
- [x] Implement validation logic in the `MonteCarloSettings` dataclass. (verify: confirm completion in repo)
- [x] Define scope for: Ensure error messages for missing or invalid fields are clear (verify: confirm completion in repo)
- [x] Implement focused slice for: Ensure error messages for missing or invalid fields are clear (verify: confirm completion in repo)
- [x] Validate focused slice for: Ensure error messages for missing or invalid fields are clear (verify: confirm completion in repo)
- [x] informative. (verify: formatter passes)
- [x] Update `src/trend_analysis/monte_carlo/__init__.py` to import and expose the newly implemented `MonteCarloScenario` and `MonteCarloSettings` classes.
- [x] Add or update unit tests to explicitly test the loading and validation of `config/scenarios/monte_carlo/example.yml`, including both successful loads and error scenarios for invalid configurations.
- [x] Define scope for: Add unit tests to test successful loading of `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)
- [x] Implement focused slice for: Add unit tests to test successful loading of `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)
- [x] Validate focused slice for: Add unit tests to test successful loading of `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)
- [x] Define scope for: Add unit tests to test error scenarios for invalid configurations in `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)
- [x] Implement focused slice for: Add unit tests to test error scenarios for invalid configurations in `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)
- [x] Validate focused slice for: Add unit tests to test error scenarios for invalid configurations in `config/scenarios/monte_carlo/example.yml`. (verify: tests pass)

## Acceptance Criteria
- [x] A valid example scenario file exists at `config/scenarios/monte_carlo/example.yml` and can be successfully loaded and validated by the application.
- [x] The scenario file, including the new example file, must be parsed correctly; any missing or invalid fields should trigger clear, specific error messages during validation.
- [x] The `MonteCarloScenario` and `MonteCarloSettings` dataclasses must include complete and updated docstrings that fully document the schema, including all required fields and constraints.
- [x] Implement the `MonteCarloScenario` dataclass and `MonteCarloSettings` dataclass with full validation logic and expose them in `src/trend_analysis/monte_carlo/__init__.py`.
- [x] Unit tests explicitly test the loading and validation of `config/scenarios/monte_carlo/example.yml`, including both successful loads and error scenarios for invalid configurations.
