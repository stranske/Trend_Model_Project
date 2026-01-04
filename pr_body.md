<!-- pr-preamble:start -->
> **Source:** Issue #4178

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Before building any natural language (NL) layer that modifies configuration, we must understand the exact config schema, validation mechanisms, and execution entrypoints. Without this foundation, the NL layer risks targeting unstable internal interfaces, producing invalid configs, or bypassing critical validation.

This is a **blocker** for all subsequent NL integration work.

#### Tasks
- [x] Trace YAML config loading path from CLI/Streamlit entry to internal use
- [x] Document validation mechanism (Pydantic models, dataclasses, custom validators, or none)
- [x] Enumerate all top-level config sections and their purposes
- [x] Classify each config key as:
- [x] - **Safe**: Can be freely modified via NL (e.g., `top_n`, `target_vol`)
- [x] - **Constrained**: Requires validation bounds (e.g., `max_weight` 0-1)
- [x] - **Derived**: Computed internally, should not be set directly
- [x] - **Internal**: Implementation detail, never expose to NL
- [x] Document `pipeline.run()` and `pipeline.run_full()` entrypoint signatures
- [x] Document `run_from_config()` and `run_full_from_config()` in pipeline_entrypoints.py
- [x] Identify any config preprocessing or normalization steps
- [x] Create `docs/planning/nl-config-audit.md` with findings

#### Acceptance criteria
- [x] `docs/planning/nl-config-audit.md` exists and documents:
- [x] - Config load path (file → parse → validate → use)
- [x] - Validation mechanism and where it runs
- [x] - Canonical schema source (`config/defaults.yml` vs models)
- [x] - Run entrypoint signatures with parameter documentation
- [x] - Classification of at least 20 config keys
- [x] Document is reviewed and linked from main NL planning doc

<!-- auto-status-summary:end -->
