<!-- pr-preamble:start -->
> **Source:** Issue #4146

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The project maintains two config truths:
1. **Lightweight schema** (`src/trend/config_schema.py`) - For fast CLI/Streamlit startup validation
2. **Full Pydantic model** (`src/trend_analysis/config/models.py`) - For complete configuration with defaults

This architecture is intentional (fast-fail without loading full Pydantic dependency chain), but creates risk of desync. The existing `tests/test_config_alignment.py` partially mitigates this, but doesn't catch:
- Config keys that are read but never validated
- Config keys that are validated but never used
- Runtime divergence when fallback behavior activates

#### Tasks
- [x] Create `src/trend_analysis/config/coverage.py` with `ConfigCoverageTracker` class
- [x] Add `track_read(key: str)` method called when config values are accessed
- [x] Add `track_validated(key: str)` method called when values pass schema validation
- [x] Add `generate_report() -> ConfigCoverageReport` returning read/validated/ignored sets
- [x] Instrument `validate_core_config()` to track validated keys
- [x] Add optional `--config-coverage` flag to CLI that dumps report after run
- [x] Add `coverage_report` field to `DiagnosticResult` when running in debug mode
- [x] Extend `tests/test_config_alignment.py` to verify coverage report catches known gaps
- [ ] Add CI job that runs config coverage on demo config and fails if ignored keys > threshold (needs-human: workflow edit required)

#### Acceptance criteria
- [x] `ConfigCoverageTracker` correctly tracks read vs validated keys
- [x] CLI `--config-coverage` flag produces human-readable report
- [x] Report identifies keys in schema but never read (potential dead config)
- [x] Report identifies keys read but not in schema (potential validation gap)
- [x] Integration test validates report catches intentionally misaligned key
- [ ] Demo config coverage shows 0 ignored keys

<!-- auto-status-summary:end -->
