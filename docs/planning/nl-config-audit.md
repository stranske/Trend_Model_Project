# NL Config Audit

Purpose: map the current YAML configuration flow, validation, and runtime entrypoints so NL-driven config changes can target stable, validated surfaces.

## Config Load Paths (file -> parse -> validate -> use)

### CLI (trend-model)
- Entry: `src/trend_analysis/cli.py` (`trend-model run`).
- Load: `load_config(args.config)` in `src/trend_analysis/config/models.py`.
  - Reads YAML via `yaml.safe_load`, validates required top-level dict sections, then calls `validate_trend_config` (Pydantic-backed) when available.
  - Returns `Config` (Pydantic or fallback) object.
- Use: `run_simulation(cfg, df)` in `src/trend_analysis/api.py`, which dispatches to pipeline logic (`trend_analysis/pipeline_entrypoints.py` and `trend_analysis/pipeline.py`).

### CLI (trend)
- Entry: `src/trend/cli.py` (`trend run`).
- Load:
  - `_load_configuration` calls `load_core_config(cfg_path)` from `src/trend/config_schema.py` for lightweight validation.
  - Then calls `load_config(cfg_path)` from `src/trend_analysis/config/models.py` for full config validation and object creation.
  - `ensure_run_spec` from `src/trend_model/spec.py` attaches run spec metadata.
- Use: `run_simulation(cfg, returns)` from `src/trend_analysis/api.py`.

### Streamlit app
- Entry: `streamlit_app/app.py` (launched by `trend-model gui` or `trend app`).
- YAML presets:
  - Demo runner loads YAML presets from `config/presets/*.yml` via `streamlit_app/components/demo_runner.py` (`load_preset_config`).
- Guardrails (startup validation):
  - `streamlit_app/components/guardrails.py` builds a minimal payload via `streamlit_app/config_bridge.py` -> `src/trend_analysis/config/bridge.py`.
  - Validation uses `trend.config_schema.validate_core_config` (dataclasses + custom validators).
- Full config:
  - `streamlit_app/components/analysis_runner.py` builds a `Config` object directly (not YAML) from UI state and passes it to `run_simulation`.

## Validation Mechanisms

- Full schema validation: `src/trend_analysis/config/models.py` uses Pydantic when available (via `Config` model) and falls back to `SimpleBaseModel` with minimal checks.
- Minimal startup validation: `src/trend_analysis/config/model.py` defines `TrendConfig` (Pydantic) for early startup checks and path validation.
- Lightweight guardrails: `src/trend/config_schema.py` implements `CoreConfig` dataclasses with custom validators for CSV paths, frequency, and cost fields.
- Streamlit bridge: `src/trend_analysis/config/bridge.py` validates minimal payloads using `validate_core_config` and normalizes values for the UI.

## Canonical Schema Source

- Primary schema + defaults live in `config/defaults.yml` (full config example) and the `Config` model in `src/trend_analysis/config/models.py`.
- `src/trend_analysis/config/model.py` and `src/trend/config_schema.py` are partial schemas used for fast validation only; they are not the complete schema source of truth.

## Top-Level Config Sections

- `version`: schema version string.
- `data`: input paths, date/price columns, frequency, missing data handling.
- `preprocessing`: deduping, winsorization, resampling, missing-data policy overrides.
- `vol_adjust`: volatility targeting settings for scaling returns.
- `sample_split`: in/out sample window configuration.
- `portfolio`: selection, weighting, constraints, robustness, transaction cost settings.
- `benchmarks`: mapping of benchmark labels to columns.
- `regime`: market regime detection controls.
- `metrics`: performance and ranking metrics definitions.
- `export`: output formats and workbook details.
- `performance`: caching and performance logging settings.
- `run`: log level, cache directory, deterministic mode.
- `multi_period`: multi-period scheduling and constraints.
- `checkpoint_dir`: output directory for checkpoints.
- `jobs`: parallelism hint.
- `seed`: random seed for deterministic runs.

## Entrypoint Signatures

- `pipeline.run(cfg: Any) -> pd.DataFrame` in `src/trend_analysis/pipeline.py`.
  - Expects a config object or mapping compatible with `Config`.
  - Returns out-of-sample metrics frame, with diagnostics attached in `DataFrame.attrs` when present.
- `pipeline.run_full(cfg: Any) -> PipelineResult` in `src/trend_analysis/pipeline.py`.
  - Returns `PipelineResult` with payload, diagnostic, and metadata.
- `run_from_config(cfg: Any, *, bindings: ConfigBindings) -> pd.DataFrame` in `src/trend_analysis/pipeline_entrypoints.py`.
  - Loads CSV from `cfg.data.csv_path`, resolves sample split, builds `RiskStatsConfig`, and dispatches analysis.
- `run_full_from_config(cfg: Any, *, bindings: ConfigBindings) -> PipelineResult` in `src/trend_analysis/pipeline_entrypoints.py`.
  - Same inputs as `run_from_config`, but returns the full diagnostics payload.

## Config Preprocessing / Normalization

- Legacy export normalization: `src/trend_analysis/config/models.py` merges legacy `output` into `export` (formats + directory/filename).
- Wrapper unwrapping: `src/trend_analysis/pipeline_helpers.py` unwraps `__cfg__` wrappers (used by config coverage tooling).
- Missing-policy fallbacks: `src/trend_analysis/pipeline_entrypoints.py` maps `nan_policy/nan_limit` to `missing_policy/missing_limit` when present.
- Target vol normalization: `src/trend_analysis/pipeline_helpers.py` returns `None` when `vol_adjust.enabled` is false or target is invalid.
- Signal normalization: `src/trend_analysis/pipeline_helpers.py` merges `signals` settings with aliases (`trend_window`, `trend_lag`, etc.) and vol defaults.
- Sample split resolution: `src/trend_analysis/pipeline_helpers.py` converts `sample_split` settings into explicit `in_*`/`out_*` boundaries.
- Preset mutation: `src/trend_analysis/presets.py` can mutate `signals`, `vol_adjust`, and `run.trend_preset` based on named presets.

## Config Key Classification (NL Safety)

Legend:
- Safe: can be modified freely.
- Constrained: must enforce bounds or allowed values.
- Derived: computed internally; NL should not set directly.
- Internal: implementation detail; never expose to NL.

| Key | Class | Notes |
| --- | --- | --- |
| `data.csv_path` | Constrained | Must exist; validated by `TrendConfig`/`CoreConfig`.
| `data.managers_glob` | Constrained | Must match CSVs; glob validation in `TrendConfig`/`CoreConfig`.
| `data.date_column` | Safe | String column name; used for parsing.
| `data.frequency` | Constrained | Allowed: D/W/M/ME (validated).
| `data.missing_policy` | Constrained | Allowed values (drop/ffill/zero).
| `data.missing_limit` | Constrained | Non-negative int or mapping.
| `data.risk_free_column` | Safe | Optional string; validated in selection stage.
| `data.allow_risk_free_fallback` | Safe | Boolean toggle for heuristic fallback.
| `preprocessing.de_duplicate` | Safe | Boolean.
| `preprocessing.winsorise.enabled` | Safe | Boolean.
| `preprocessing.winsorise.limits` | Constrained | Two floats between 0 and 1.
| `preprocessing.resample.target` | Constrained | Supported frequency target (D/W/M/ME or None).
| `preprocessing.missing_data.policy` | Constrained | Allowed values (drop/ffill/zero).
| `preprocessing.missing_data.limit` | Constrained | Non-negative int or None.
| `vol_adjust.enabled` | Safe | Boolean.
| `vol_adjust.target_vol` | Constrained | Float > 0 to enable scaling.
| `vol_adjust.window.length` | Constrained | Positive int.
| `vol_adjust.window.decay` | Constrained | Allowed values: ewma/simple.
| `vol_adjust.window.lambda` | Constrained | 0 < lambda < 1 for ewma.
| `vol_adjust.floor_vol` | Constrained | Float >= 0.
| `vol_adjust.warmup_periods` | Constrained | Non-negative int.
| `sample_split.method` | Constrained | Allowed values: date/ratio.
| `sample_split.date` | Constrained | YYYY-MM or YYYY-MM-DD depending on parser.
| `sample_split.ratio` | Constrained | 0 < ratio < 1.
| `portfolio.selection_mode` | Constrained | Allowed values: all/random/manual.
| `portfolio.random_n` | Constrained | Positive int.
| `portfolio.manual_list` | Safe | List of fund column names.
| `portfolio.weighting_scheme` | Constrained | Allowed values: equal/risk_parity/hrp/etc.
| `portfolio.rebalance_freq` | Constrained | Allowed values: M/Q/A/None.
| `portfolio.transaction_cost_bps` | Constrained | Non-negative float.
| `portfolio.max_turnover` | Constrained | Float between 0 and 1.
| `portfolio.rank.inclusion_approach` | Constrained | Allowed: top_n/top_pct/threshold.
| `portfolio.rank.n` | Constrained | Positive int.
| `portfolio.rank.pct` | Constrained | 0 < pct <= 1.
| `portfolio.rank.threshold` | Constrained | Positive float (z-score threshold).
| `portfolio.rank.bottom_k` | Constrained | Non-negative int.
| `portfolio.rank.score_by` | Constrained | Allowed metric names (registry).
| `portfolio.rank.blended_weights` | Constrained | Weights sum to 1.
| `portfolio.constraints.long_only` | Safe | Boolean.
| `portfolio.constraints.max_weight` | Constrained | 0 < max_weight <= 1.
| `portfolio.constraints.group_caps` | Constrained | Values 0-1 per group.
| `portfolio.robustness.shrinkage.method` | Constrained | Allowed: none/ledoit_wolf/oas.
| `benchmarks` | Safe | Mapping label -> column.
| `metrics.registry` | Constrained | Must map to known metrics for RiskStatsConfig.
| `metrics.rf_rate_annual` | Constrained | Float >= 0.
| `export.formats` | Constrained | Known format list (xlsx/csv/json/txt).
| `performance.enable_cache` | Safe | Boolean toggle.
| `run.log_level` | Constrained | DEBUG/INFO/WARNING/ERROR.
| `run.trend_preset` | Derived | Written by preset application, not user-editable.
| `multi_period.frequency` | Constrained | Allowed: M/Q/A.
| `seed` | Safe | Integer seed, overrides randomness.
| `__cfg__` | Internal | Wrapper key for config coverage tooling.
| `output.*` | Derived | Legacy conversion into `export` during load.

## Notes for NL Layer

- NL modifications should target `Config`-level fields that are validated by `config/models.py` and enforce bounds for constrained fields.
- Avoid `__cfg__` and `output` keys entirely; they are internal/legacy.
- When in doubt, validate with `trend_analysis.config.model.validate_trend_config` or `trend.config_schema.validate_core_config` depending on scope.
