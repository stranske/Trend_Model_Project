<!-- bootstrap for codex on issue #3645 -->

## Scope

- Harden config handling so typos in keys or wrong value types are caught before the runtime pipeline explodes.
- Keep the schema definition in a single shared module that both the CLI and application code import so behaviour never drifts.
- Stick to stdlib/dataclasses plus handwritten validation so we do not balloon dependencies or break CI expectations.

## Task list

- [x] Define a lightweight config schema for the core knobs (data paths, universe file, frequency, costs, etc.). Implemented in
  `src/trend_analysis/config/model.py` via the `TrendConfig`/`DataSettings`/`PortfolioSettings`/`RiskSettings` models that both
  the CLI and application share.
- [x] Validate configs on startup, including defaults and helpful error text for missing/invalid values. This is provided by
  `load_trend_config()`/`validate_trend_config()` which raise descriptive `ValueError`s for bad paths, unsupported frequencies,
  or malformed cost/risk settings.
- [x] Add tests that cover missing required fields and wrong types so regressions fail quickly. The scenarios live under
  `tests/test_trend_config_model.py` and `tests/test_trend_config_model_negative_paths.py`, exercising missing fields, invalid
  paths, and type mismatches.

## Acceptance criteria

- [x] Invalid configs fail fast with a single, clear error message. Tests such as
  `tests/test_trend_config_model.py::test_trend_config_rejects_invalid_frequency` and the negative-path suite confirm the
  validator surfaces the first offending field with actionable text.
- [x] Valid configs round-trip through load + validate without mutation or data loss. Covered by
  `tests/test_trend_config_model.py::test_load_trend_config_defaults` and
  `tests/test_trend_config_model.py::test_validate_trend_config_locates_csv_relative_to_parent`, which write configs to disk,
  reload them, and ensure the resolved values are preserved.
