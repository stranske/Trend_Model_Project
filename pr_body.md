<!-- pr-preamble:start -->
> **Source:** Issue #4178

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Before building any natural language (NL) layer that modifies configuration, we must understand the exact config schema, validation mechanisms, and execution entrypoints. Without this foundation, the NL layer risks targeting unstable internal interfaces, producing invalid configs, or bypassing critical validation.

This is a **blocker** for all subsequent NL integration work.

#### Tasks
- [ ] Trace YAML config loading path from CLI/Streamlit entry to internal use
- [ ] Document validation mechanism (Pydantic models, dataclasses, custom validators, or none)
- [ ] Enumerate all top-level config sections and their purposes
- [ ] Classify each config key as:
- [ ] - **Safe**: Can be freely modified via NL (e.g., `top_n`, `target_vol`)
- [ ] - **Constrained**: Requires validation bounds (e.g., `max_weight` 0-1)
- [ ] - **Derived**: Computed internally, should not be set directly
- [ ] - **Internal**: Implementation detail, never expose to NL
- [ ] Document `pipeline.run()` and `pipeline.run_full()` entrypoint signatures
- [ ] Document `run_from_config()` and `run_full_from_config()` in pipeline_entrypoints.py
- [ ] Identify any config preprocessing or normalization steps
- [ ] Create `docs/planning/nl-config-audit.md` with findings

#### Acceptance criteria
- [ ] `docs/planning/nl-config-audit.md` exists and documents:
- [ ] - Config load path (file → parse → validate → use)
- [ ] - Validation mechanism and where it runs
- [ ] - Canonical schema source (`config/defaults.yml` vs models)
- [ ] - Run entrypoint signatures with parameter documentation
- [ ] - Classification of at least 20 config keys
- [ ] Document is reviewed and linked from main NL planning doc

**Head SHA:** 21866bf8bc36a7c8472758296ec2075989531008
**Latest Runs:** ⏹️ cancelled — Gate
**Required:** gate: ⏹️ cancelled

| Workflow / Job | Result | Logs |
|----------------|--------|------|
| Agents Bot Comment Handler | ⏹️ cancelled | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983844) |
| Agents Keepalive Loop | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983831) |
| Agents PR Meta | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983963) |
| Autofix | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983861) |
| Copilot code review | ⏳ queued | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692984352) |
| Gate | ⏹️ cancelled | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983847) |
| Health 45 Agents Guard | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983833) |
| PR 11 - Minimal invariant CI | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983887) |
| PR 12 - Fund selector Playwright smoke | ❔ in progress | [View run](https://github.com/stranske/Trend_Model_Project/actions/runs/20692983894) |
<!-- auto-status-summary:end -->
