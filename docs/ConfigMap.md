# Configuration Map

This document catalogues configuration and environment templates in the repository, notes where they are consumed in code, and flags any retired files.

## Pipeline and analysis configs
| Path | Purpose | Primary consumers |
| --- | --- | --- |
| `config/defaults.yml` | Baseline schema that loaders fall back to when no config path is provided. | `src/trend_analysis/config/legacy.py` default pointer; discovery logic in `src/trend_analysis/config/models.py`. |
| `config/demo.yml` | End-to-end demo scenario used by CI and helper scripts. | Loaded in `scripts/run_multi_demo.py`, `scripts/generate_demo.py`, `scripts/run_threshold_churn_demo.py`, and `scripts/demo_os_summary.py`. |
| `config/portfolio_test.yml` | Portfolio-selection baseline for debugging and regression tests. | Referenced by `examples/debug_fund_selection.py` and `tests/test_multi_period_selection.py`. |
| `config/trend.toml` | TOML example for the `trend-run` CLI entry point. | Exercised in `tests/test_spec_loader.py` and documented in `docs/CLI.md`. |
| `config/robust_demo.yml` | Scenario for exercising robustness controls. | Referenced in `ROBUSTNESS_GUIDE.md` test steps. |
| `config/long_backtest.yml` | Default config for the real-model runner. | Default argument in `scripts/run_real_model.py`. |
| `config/walk_forward.yml` | Walk-forward/grid search example. | Default argument in `scripts/walk_forward.py`. |
| `config/presets/*.yml` | Risk-profile presets exposed to CLI/Streamlit users. | Documented in `docs/PresetStrategies.md` and `docs/UserGuide.md`. |
| `config/trend_universe_2004.yml` | Legacy trend-universe sample kept for reproducible docs examples. | Mentioned in `README.md`; no active loader references. |
| `config/trend_concentrated_2004.yml` | Concentrated trend sample retained for manual runs. | No active code references; keep aligned with other trend configs. |
| `config/universe/*.yml` | Standalone universe definitions for manual experimentation. | Mentioned in `README.md`; not loaded directly by current scripts. |

## Quality and tooling configs
| Path | Purpose | Primary consumers |
| --- | --- | --- |
| `tests/quarantine.yml` | Quarantine policy definitions for CI linting. | Validated by `tools/validate_quarantine_ttl.py` and related tests. |
| `requirements.lock` | Pinned dependency set for reproducible environments. | Synced/consumed by environment and quality scripts plus the CLI. |
| `pyproject.toml` | Build, lint, and dependency source of truth. | Parsed/used by quality gate and dependency sync scripts. |
| `docker-compose.yml` | API server/docker runtime wiring. | Port alignment noted in `src/trend_analysis/api_server/__main__.py`. |
| `cliff.toml` | Changelog formatting rules. | Referenced in `docs/release-process.md`. |

## Archived templates

The retired configs below live in `archives/config/`:

| Deprecated file | Replacement guidance |
| --- | --- |
| `archives/config/rolling_hold_bayes.yml` | Use `config/walk_forward.yml` for maintained multi-period runs instead of the legacy threshold-hold workflow. |
| `archives/config/cv_example.yml` | Use `config/walk_forward.yml` for cross-validation/grid search or `config/long_backtest.yml` for full backtests. |

## Notes
- The `docs/` tree currently contains no standalone `*.yml`, `*.env.example`, or `*.toml` files; config guidance lives in prose docs instead.
- Keep this map updated whenever configs move or new consumers are added to prevent drift.
