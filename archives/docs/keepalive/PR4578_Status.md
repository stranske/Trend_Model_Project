# Keepalive Status — PR #4578

## Scope
Add a scenario registry system that allows Monte Carlo scenarios to be discovered by name without hardcoding paths.

## Tasks
- [x] Create `config/scenarios/monte_carlo/index.yml` registry entries.
- [x] Implement `list_scenarios` function in `src/trend_analysis/monte_carlo/registry.py` to list all registered scenarios, optionally filtered by tags.
- [x] Implement `load_scenario` function in `src/trend_analysis/monte_carlo/registry.py` to load a scenario by name from the registry.
- [x] Implement `get_scenario_path` function in `src/trend_analysis/monte_carlo/registry.py` to retrieve the file path for a scenario by name.
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate basic functionality.
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate basic functionality. (verify: tests pass)
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate basic functionality. (verify: tests pass)
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate basic functionality. (verify: tests pass)
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate filtering by tags. (verify: tests pass)
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate filtering by tags. (verify: tests pass)
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate filtering by tags. (verify: tests pass)
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `list_scenarios` to validate filtering by tags.
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate the structure of the returned `MonteCarloScenario`.
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate the structure of the returned `MonteCarloScenario`. (verify: tests pass)
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate the structure of the returned `MonteCarloScenario`. (verify: tests pass)
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate the structure of the returned `MonteCarloScenario`. (verify: tests pass)
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate error handling for invalid scenarios. (verify: tests pass)
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate error handling for invalid scenarios. (verify: tests pass)
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate error handling for invalid scenarios. (verify: tests pass)
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `load_scenario` to validate error handling for invalid scenarios.
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify correct file paths for valid scenarios.
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify correct file paths for valid scenarios.
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify correct file paths for valid scenarios.
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify correct file paths for valid scenarios.
- [x] Define scope for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify error handling for missing or invalid scenarios.
- [x] Implement focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify error handling for missing or invalid scenarios.
- [x] Validate focused slice for: Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify error handling for missing or invalid scenarios.
- [x] Write unit tests in `tests/monte_carlo/test_registry.py` for `get_scenario_path` to verify error handling for missing or invalid scenarios.

## Acceptance Criteria
- [x] Scenarios can be discovered by name.
- [x] Scenarios can be filtered by tags.
- [x] Adding a new scenario file only requires editing `config/scenarios/monte_carlo/index.yml`.
- [x] `load_scenario("name")` returns a validated `MonteCarloScenario`.
- [x] Missing scenarios produce clear error messages.
- [x] Unit tests for `list_scenarios`, `load_scenario`, and `get_scenario_path` pass.

Last verified: 2026-01-29 (pytest tests/monte_carlo/test_registry.py -m "not slow")
