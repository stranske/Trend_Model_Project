# Consolidate duplicate config_bridge modules

## Why

There are two nearly identical config bridge implementations:
- `streamlit_app/config_bridge.py` - Streamlit-specific bridge
- `src/trend_analysis/config/bridge.py` - Core library bridge

Both provide `build_config_payload()` and `validate_payload()` functions with similar but subtly different implementations. This duplication creates the classic "UI says it ran X; CLI ran Y" problem when the bridges drift apart.

Key differences that can cause divergent behavior:
1. `streamlit_app/config_bridge.py` includes `"version": "1"` in payload; core bridge doesn't
2. Core bridge builds `data` dict conditionally (only includes non-None fields); Streamlit always includes all fields
3. Core bridge includes additional `cost_model` fields (`per_trade_bps`, `half_spread_bps`)

## Scope

Consolidate to a single config bridge implementation in the core library that Streamlit imports.

## Non-Goals

- Refactoring the underlying config schema validation
- Changing the public API signatures
- Modifying CLI config loading

## Tasks

- [ ] Audit differences between the two `build_config_payload()` implementations and create unified version
- [ ] Audit differences between the two `validate_payload()` implementations and create unified version
- [ ] Update `src/trend_analysis/config/bridge.py` to be the canonical implementation
- [ ] Update `streamlit_app/config_bridge.py` to import from `trend_analysis.config.bridge`
- [ ] Add deprecation warning in `streamlit_app/config_bridge.py` for direct usage
- [ ] Add unit test that imports from both locations and asserts they return identical types
- [ ] Remove duplicated code from `streamlit_app/config_bridge.py` (keep only re-exports)

## Acceptance Criteria

- [ ] Only one implementation of `build_config_payload()` and `validate_payload()` exists
- [ ] `streamlit_app.config_bridge` re-exports from `trend_analysis.config.bridge`
- [ ] All Streamlit tests pass with the consolidated bridge
- [ ] CLI and Streamlit produce identical config payloads for the same inputs
- [ ] Type signatures are consistent across both import paths

## Implementation Notes

Files to modify:
- `src/trend_analysis/config/bridge.py` - Canonical implementation
- `streamlit_app/config_bridge.py` - Thin re-export wrapper
- `tests/test_config_bridge_parity.py` - New test file

The core bridge at `src/trend_analysis/config/bridge.py` (lines 13-55) is more complete and should be the base. The Streamlit bridge's `"version": "1"` field should be incorporated if needed.
