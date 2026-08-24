# 📂 `config/` — Configuration Directory

> **Purpose:** YAML configuration files for analysis runs  
> **Last updated:** November 2025

---

## 📄 Configuration Files

| File | Description |
|------|-------------|
| `defaults.yml` | Default configuration values |
| `demo.yml` | Demo run configuration |
| `single_period.yml` | Single-period phase-3 run configuration |
| `portfolio_test.yml` | Portfolio testing configuration |
| `trend_universe_2004.yml` | Trend universe 2004 configuration |
| `trend_concentrated_2004.yml` | Concentrated trend strategy |
| `long_backtest.yml` | Long backtest configuration |
| `robust_demo.yml` | Robustness testing configuration |
| `walk_forward.yml` | Walk-forward analysis settings |
| `trend.toml` | TOML-format trend settings |
| `coverage-baseline.json` | Coverage baseline for CI |

## 📁 Subdirectories

### `universe/`
Universe definition files:
- `core.yml` — Core universe definition
- `core_plus_benchmarks.yml` — Core with benchmark indices
- `managed_futures_min.yml` — Minimal managed futures universe

### `presets/`
Pre-configured analysis presets for common use cases.

---

## 🔧 Usage

```bash
# Run with specific config
PYTHONPATH=./src python -m trend.cli run -c config/demo.yml

# Use environment variable
TREND_CFG=config/defaults.yml PYTHONPATH=./src python -m trend.cli run
```

---

*See `docs/configuration.md` for full configuration reference.*
