# Configuration Reference

This document describes the configuration options for the Trend Model Project.

## Configuration Files

### File Locations

| File | Purpose |
|------|---------|
| `config/defaults.yml` | Default values (fallback) |
| `config/demo.yml` | Demo/test configuration |
| `config/presets/*.yml` | Risk profile presets |
| `config/universe/*.yml` | Universe definitions |

### Loading Priority

1. Command-line argument: `-c path/to/config.yml`
2. Environment variable: `TREND_CFG=path/to/config.yml`
3. Default: `config/defaults.yml`

## Minimal configuration layers

Two lightweight schemas validate configuration long before the full analysis
model is constructed:

- `trend.config_schema.CoreConfig` is a stdlib-only contract used by the CLI
  and Streamlit app to fail fast on missing CSV inputs, invalid frequencies, or
  obviously broken cost settings. It resolves data paths relative to the config
  file, normalises them to absolute paths, and defaults `cost_model` fields to
  the top-level `transaction_cost_bps` when explicit values are not provided.
- `trend_analysis.config.model.TrendConfig` is a pared-down Pydantic model
  enforced at runtime. It re-validates the same data/portfolio knobs while also
  covering runtime-only options such as calendars, risk targets, and turnover
  controls.

### Invariants to keep aligned

- `data.csv_path` or `data.managers_glob` **must** point to existing files when
  resolved relative to the config location; both schemas reject missing
  matches.
- `data.date_column` and `data.frequency` are normalised to non-empty strings
  with the same allowed frequency values (`D`, `W`, `M`, `ME`).
- `data.universe_membership_path`, when provided, is resolved to an existing
  file in both layers.
- `portfolio.transaction_cost_bps` and every linear `cost_model` field (bps per
  trade, slippage, per-trade bps, half-spread) remain non-negative and must
  round-trip unchanged through `CoreConfig.to_payload()` into `TrendConfig`.

### Intentional differences

- `TrendConfig` requires runtime controls that the minimal core layer ignores
  (for example `portfolio.rebalance_calendar`, turnover caps, and
  `vol_adjust` settings). Changes to those fields do not affect the core
  contract.
- Risk-free column, missing-data policies, and other data-cleaning fields are
  only enforced in `TrendConfig`; the core schema leaves them unchecked so the
  CLI/UI stays lean.
- `TrendConfig` forbids glob characters in `data.csv_path` while
  `data.managers_glob` explicitly accepts globs; `CoreConfig` mirrors the same
  behaviour but reports errors using stdlib helpers instead of Pydantic.

## Configuration Sections

### Data Section

```yaml
data:
  csv_path: demo/demo_returns.csv
  date_column: Date
  columns: null  # null = all columns except date
  missing_policy: drop  # drop, ffill, zero
  missing_fill_limit: 3  # max consecutive fills
```

### Portfolio Section

```yaml
portfolio:
  top_n: 8                    # Number of assets to select
  lookback: 12                # Months for trend calculation
  rebalance_frequency: M      # M=monthly, Q=quarterly
  weighting: equal            # equal, score_prop, vol_adjusted
  
  # Selection options
  selection:
    mode: rank                # all, random, manual, rank
    score_by: Sharpe          # Sharpe, AnnualReturn, blended
```

### Risk Section

```yaml
risk:
  vol_target: 0.10            # Target volatility (10%)
  max_position: 0.25          # Max 25% per position
  min_position: 0.02          # Min 2% per position
```

### Output Section

```yaml
output:
  format: excel               # csv, json, excel
  path: outputs/analysis      # Output directory/prefix
  include_raw_metrics: true   # Include detailed metrics
```

### Walk-Forward Section

```yaml
walk_forward:
  train: 36                   # In-sample months
  test: 12                    # Out-of-sample months
  step: 6                     # Step size
```

## Preset Configurations

Pre-built risk profiles in `config/presets/`:

| Preset | Description |
|--------|-------------|
| `conservative.yml` | Lower risk, stable returns |
| `balanced.yml` | Moderate risk/return |
| `aggressive.yml` | Higher risk, growth focus |
| `cash_constrained.yml` | Limited cash allocation |

### Using Presets

```bash
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/presets/balanced.yml
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `TREND_CFG` | Default config file path |
| `PYTHONPATH` | Add `./src` for module imports |

## See Also

- [ConfigMap.md](ConfigMap.md) - Complete config file inventory
- [PresetStrategies.md](PresetStrategies.md) - Preset details
- [UserGuide.md](UserGuide.md) - User documentation
