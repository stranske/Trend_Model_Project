# Add config coverage report for schema alignment validation

## Why

The project maintains two config truths:
1. **Lightweight schema** (`src/trend/config_schema.py`) - For fast CLI/Streamlit startup validation
2. **Full Pydantic model** (`src/trend_analysis/config/models.py`) - For complete configuration with defaults

This architecture is intentional (fast-fail without loading full Pydantic dependency chain), but creates risk of desync. The existing `tests/test_config_alignment.py` partially mitigates this, but doesn't catch:
- Config keys that are read but never validated
- Config keys that are validated but never used
- Runtime divergence when fallback behavior activates

## Scope

Generate a runtime "config coverage report" that lists which config keys were read, which were validated, and which were ignored during a pipeline run.

## Non-Goals

- Changing the two-schema architecture
- Making the lightweight schema feature-complete with Pydantic
- Runtime performance optimization

## Tasks

- [ ] Create `src/trend_analysis/config/coverage.py` with `ConfigCoverageTracker` class
- [ ] Add `track_read(key: str)` method called when config values are accessed
- [ ] Add `track_validated(key: str)` method called when values pass schema validation
- [ ] Add `generate_report() -> ConfigCoverageReport` returning read/validated/ignored sets
- [ ] Instrument `validate_core_config()` to track validated keys
- [ ] Add optional `--config-coverage` flag to CLI that dumps report after run
- [ ] Add `coverage_report` field to `DiagnosticResult` when running in debug mode
- [ ] Extend `tests/test_config_alignment.py` to verify coverage report catches known gaps
- [ ] Add CI job that runs config coverage on demo config and fails if ignored keys > threshold

## Acceptance Criteria

- [ ] `ConfigCoverageTracker` correctly tracks read vs validated keys
- [ ] CLI `--config-coverage` flag produces human-readable report
- [ ] Report identifies keys in schema but never read (potential dead config)
- [ ] Report identifies keys read but not in schema (potential validation gap)
- [ ] Integration test validates report catches intentionally misaligned key
- [ ] Demo config coverage shows 0 ignored keys

## Implementation Notes

Files to create:
- `src/trend_analysis/config/coverage.py` - Tracker implementation

Files to modify:
- `src/trend/config_schema.py` - Instrument `validate_core_config()`
- `src/trend_analysis/cli.py` - Add `--config-coverage` flag
- `src/trend_analysis/diagnostics.py` - Add coverage to diagnostic payload
- `tests/test_config_alignment.py` - Add coverage report tests

Example report format:
```
Config Coverage Report
======================
Keys validated: 45
Keys read: 52
Keys ignored: 3
  - portfolio.legacy_turnover_mode (read but not validated)
  - data.deprecated_format (validated but never read)
  - vol_adjust.experimental_scaling (neither validated nor read)
```

The existing manual debugging approach in `docs/validation/ui_run_2025-12-15.md` shows this is already a real pain point during debugging sessions.
