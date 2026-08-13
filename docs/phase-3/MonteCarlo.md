# Phase 3: Monte Carlo Framework

> **Status**: Design Complete | Implementation Complete<br>
> **Target**: Multi-path simulation for portfolio forecasting and risk quantification

---

## Table of Contents

1. [Overview](#overview)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [Architecture](#architecture)
4. [Return Generation Models](#return-generation-models)
5. [Strategy Sets](#strategy-sets)
6. [Cost and Turnover Modeling](#cost-and-turnover-modeling)
7. [Multi-Period Integration](#multi-period-integration)
8. [Fold Support](#fold-support)
9. [Scenario Library](#scenario-library)
10. [Output Specifications](#output-specifications)
11. [CLI and Streamlit Integration](#cli-and-streamlit-integration)
12. [Implementation Milestones](#implementation-milestones)

---

## Overview

### What We're Building

A Monte Carlo simulation layer that enables:

| Capability | Priority | Description |
|------------|----------|-------------|
| **Portfolio Outcome Forecasting** | Primary | Simulate 10–50 year horizons, monthly frequency (daily optional) |
| **Drawdown & Tail Risk** | Secondary | Distribution of Sharpe, max drawdown, tail outcomes, time-under-water |
| **Construction Uncertainty** | Primary | Vary selection, weighting, vol targeting, turnover, constraints |
| **Return Uncertainty** | Primary | Simulated price paths preserving correlation and volatility clustering |

### Execution Modes

| Mode | Description | Default |
|------|-------------|---------|
| **Two-Layer** | Fixed strategy set evaluated on common simulated paths (variance reduction) | ✅ Yes |
| **Mixture** | Each path samples a strategy variant independently | Optional |

The two-layer mode uses **common random numbers**—the same generated price path is reused across all strategies—providing significant variance reduction for strategy comparisons.

### Variance Reduction Expectations (CRN)

**Expected factor (documented benchmark):** For the synthetic CRN test in `tests/monte_carlo/test_seed.py` (path seed 2026, `n_paths=200`, `horizon=50`, `noise_scale=0.1`), the expected variance ratio is approximately:

```
ratio ≈ noise_scale^2 / (1 + noise_scale^2) ≈ 0.0099
```

This implies an **expected variance-reduction factor of ~100x** (independent variance / common variance). The rationale: with common random numbers, the shared base path cancels when comparing strategy outputs, leaving only idiosyncratic noise; when paths are independent, both base and noise contribute to variance.

**Empirical result (2026-02-02):** Running the benchmark with the parameters above yields:

- Common variance ≈ `3.9086e-4`
- Independent variance ≈ `3.8478e-2`
- Observed ratio ≈ `0.01016` → **~98.4x reduction**

This meets the documented expectation of ~100x reduction for the benchmark configuration.

### Mixture-Mode Determinism

Mixture mode remains deterministic when a scenario seed is set, but it has two valid
execution modes:

1. **Shared-path bulk generation**: generate all paths in one bulk call, then evaluate
   one sampled strategy per path.
2. **Per-path generation**: generate each path independently with that path's seed,
   then evaluate one sampled strategy for that path.

The runner chooses between those modes from the effective per-path seeds:

- Use **shared-path bulk generation** when every non-null path seed matches the
  value that `SeedManager(<scenario seed>).get_path_seed(path_id)` would produce
  for that path. This is the default for a seeded scenario because `_build_seeds()`
  derives path seeds from the same base `SeedManager`.
- If all path seeds are `None`, also use **shared-path bulk generation**. This
  still uses one bulk model call with `seed=<monte_carlo.seed>`; when the scenario
  itself is unseeded, that bulk-call seed is `None`.
- Use **per-path generation** when any non-null per-path seed differs from the
  base `SeedManager` sequence. This can happen in tests, debug harnesses, or any
  future override that supplies explicit path seeds.

To predict the mode for a scenario, check `monte_carlo.seed` first. If it is set and
the run uses the built-in seed builder, mixture mode will use shared-path bulk
generation. If a caller overrides path seeds, compare each non-null override with
`SeedManager(seed).get_path_seed(path_id)`: one mismatch switches the run to
per-path generation. Strategy seeds do not choose the path-generation mode; they only
choose which strategy variant is sampled for each path.

Both modes are deterministic for fixed inputs and fixed seeds; they differ only in
how path generation work is scheduled. The mode-selection contract is covered by
`tests/monte_carlo/test_runner.py::test_run_mixture_uses_shared_bulk_generation_when_path_seeds_match_base`,
`test_run_mixture_uses_shared_bulk_generation_when_path_seeds_are_unset`, and
`test_run_mixture_uses_per_path_generation_when_path_seeds_diverge`.
Completed runner output exposes the selected mode in
`MonteCarloResults.metadata["path_generation_mode"]` as `shared` or `per-path`.

#### Troubleshooting and Performance Tradeoffs

If mixture mode selects **per-path generation**, the runner performs many small
model calls (`n_paths=1`) instead of one bulk call (`n_paths=N`). This can be slower
due to repeated setup overhead and reduced vectorization compared with shared-path
bulk generation. The debug log includes
`Mixture mode path generation selected mode=<shared|per-path> reason=<...>`.
Use `MonteCarloResults.metadata["path_generation_mode"]` to verify the selected mode
programmatically for troubleshooting and test assertions.

---

## Goals and Non-Goals

### Goals

- Forecast portfolio NAV distributions over extended horizons
- Quantify probability of drawdown breaches and tail events
- Compare strategy variants under identical market scenarios
- Support both curated strategy packs and sampled parameter sweeps
- Integrate with existing CLI and Streamlit interfaces
- Produce exportable output bundles (parquet/CSV/JSON)

### Non-Goals

- Real-time or intraday simulation (monthly is primary, daily optional)
- Leverage modeling (portfolios remain fully invested or under-invested)
- Complex derivative pricing or exotic instrument simulation
- Factor model return generation (bootstrap preserves empirical structure)

---

## Architecture

### Key Design Decision: Path Context + Strategy Evaluation

Efficiency comes from separating work into two phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE A: Path Context                     │
│                   (Expensive, Strategy-Agnostic)             │
├─────────────────────────────────────────────────────────────┤
│  For each Monte Carlo path:                                  │
│    1. Generate synthetic prices                              │
│    2. Preprocess and align data                              │
│    3. Build multi-period schedule                            │
│    4. Compute score frames / metric frames for all           │
│       rebalance dates (superset of all strategy needs)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE B: Strategy Evaluation                 │
│                   (Cheap, Strategy-Specific)                 │
├─────────────────────────────────────────────────────────────┤
│  For each strategy configuration:                            │
│    1. Run selection + rebalancing policy                     │
│    2. Apply weighting + constraints + costs                  │
│    3. Produce portfolio weights over time                    │
│    4. Compute portfolio-level metrics (Sharpe, maxDD, etc.)  │
└─────────────────────────────────────────────────────────────┘
```

This separation is the **core efficiency principle**: path context is computed once and reused across all strategies.

### Module Structure

```
src/trend_analysis/monte_carlo/
├── __init__.py
├── scenario.py          # Schema + loader + validation
├── registry.py          # Scenario discovery
├── models/
│   ├── __init__.py
│   ├── base.py          # PricePathModel interface
│   ├── bootstrap.py     # Stationary bootstrap
│   └── regime.py        # Regime labeling + conditioned bootstrap
├── strategy/
│   ├── __init__.py
│   ├── variant.py       # Strategy representation
│   └── sampler.py       # Parameter sampling
├── costs.py             # Stochastic cost process
├── runner.py            # MonteCarloRunner (two-layer + mixture)
├── cache.py             # Path context caching
├── aggregator.py        # Distribution computation
└── export.py            # Output bundle writer
```

---

## Return Generation Models

### Data Flow

```
Historical Prices → Log Returns → Bootstrap/Regime Model → Synthetic Returns → Synthetic Prices
```

**Why log returns?**  
- Additive in time (easier block stitching)
- Reconstituted prices are always positive
- Standard practice for return modeling

### Model 1: Multivariate Stationary Bootstrap (Primary)

The default model, offering highest realism per unit of implementation effort.

**Concept:**
1. Convert prices to log returns: $r_t = \log(P_t / P_{t-1})$
2. Generate synthetic series by stitching random-length blocks of historical returns
3. Block lengths are geometrically distributed (mean length `L`) to reduce periodic artifacts
4. Sample the **full multivariate vector** each step to preserve cross-asset correlation

**Parameters:**
```yaml
return_model:
  kind: stationary_bootstrap
  mean_block_len: 6          # Average block length in periods
  calibration_window: null   # Uses full history if null
```

**Key Properties:**
- ✅ Captures historical tail events and volatility clustering
- ✅ Preserves cross-sectional correlation structure
- ✅ Handles missingness by carrying availability masks
- ✅ Simple to implement, validate, and explain

### Model 2: Regime-Conditioned Bootstrap (Enhanced)

Adds a regime process for better drawdown realism.

**Concept:**
1. Classify historical periods into regimes (e.g., calm vs stress)
2. Estimate regime transition probabilities
3. Simulate a regime path forward
4. Bootstrap blocks **from matching regime buckets**

**Parameters:**
```yaml
return_model:
  kind: regime_conditioned_bootstrap
  mean_block_len: 6
  regime:
    enabled: true
    kind: proxy_vol_threshold   # Or: hmm, explicit_labels
    proxy_column: SPX           # Benchmark for regime detection
    threshold_percentile: 75    # Above = stress regime
```

**Key Properties:**
- ✅ Higher volatility clustering in stress periods
- ✅ Correlation spikes during stress captured
- ✅ Supports regime-dependent costs and constraints

---

## Strategy Sets

### Two Sources of Strategy Variation

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Curated Strategies | Named configurations representing distinct construction philosophies |
| **Tier 2** | Sampled Variants | Randomly generated from parameter distributions |

### Tier 1: Curated Strategies

12–30 named strategies per sleeve/asset class, covering:

| Axis | Variants |
|------|----------|
| Selection | Rank vs threshold-hold vs random stress test |
| Holding Count | Concentrated (8) vs diversified (16+) |
| Weighting | Equal vs score-proportional vs shrinkage/Bayes |
| Vol Target | On/off + target level |
| Turnover | Tight (8%) vs moderate (15%) vs loose (25%) |
| Constraints | Max weight caps, max active positions, cash handling |

**Example:**
```yaml
curated:
  - name: Rank_12_Equal_TightTurnover
    overrides:
      portfolio:
        selection_mode: rank
        rank:
          n: 12
          metric: Sharpe
        weighting_scheme: equal
        max_turnover: 0.08
        
  - name: ThresholdHold_16_Bayes_ModerateTurnover
    overrides:
      portfolio:
        selection_mode: rank
        rank:
          inclusion_approach: threshold
          threshold: 0.3
          n: 16
        weighting_scheme: shrinkage
        max_turnover: 0.15
```

### Tier 2: Sampled Variants

Draw strategy configs from distributions:

```yaml
sampled:
  enabled: true
  n_strategies: 100
  sampling:
    portfolio.rank.n:
      dist: categorical
      values: [8, 12, 16, 20]
    portfolio.max_turnover:
      dist: uniform
      low: 0.08
      high: 0.25
    portfolio.constraints.max_weight:
      dist: categorical
      values: [0.15, 0.20, 0.25, null]
```

**Constraints on sampling:**
- Reject invalid combinations (e.g., n_holdings > universe size)
- Limit rejection attempts to avoid infinite loops
- Log rejected configs for debugging

---

## Cost and Turnover Modeling

### Cash Handling

**Principle:** Underinvestment is explicit, not hidden.

```
Portfolio Weights = Asset Weights + Cash Weight = 1.0
Cash Earns = Risk-Free Rate (configurable source)
```

**Implementation:**
- If turnover cap, drawdown guard, or constraints cause underinvestment → residual goes to CASH
- RF source options: explicit series, constant rate, benchmark proxy
- Portfolio return = weighted sum including cash component

### CASH Injection Policy

- Trigger condition (boolean):
  - `inject_cash = bool(metrics.rf_override_enabled)`.
- `data.risk_free_column` and `data.allow_risk_free_fallback` only choose or discover a risk-free series for metric calculations. They do not bypass the CASH injection gate.
- Injected CASH rate:
  - If `metrics.rf_override_enabled` is `true`, use `metrics.rf_rate_annual`.
  - Convert to periodic simulation rate with `periods_per_year_from_code(data.frequency)`.
  - Formula: `rf_periodic = (1 + rf_rate_annual) ** (1 / periods_per_year) - 1`.
- Skip behavior: if `metrics.rf_override_enabled` is disabled, no `CASH` column is injected (even when `data.risk_free_column` or `data.allow_risk_free_fallback` is set).
- If `metrics.rf_override_enabled` is enabled but risk-free resolution fails, resolves to an unsupported type, or yields a series that cannot align cleanly to the simulated returns, the runner logs one warning and leaves the return frame unchanged.
- If a `CASH` or legacy `cash` column is already present, the runner preserves it and does not overwrite the existing values.

Example:
- Config sets `data.frequency: "M"`, `metrics.rf_override_enabled: true`, and `metrics.rf_rate_annual: 0.12`.
- Then `rf_periodic = (1.12) ** (1/12) - 1 ≈ 0.009489`.
- For a simulated return frame with no `CASH` column, the runner injects `CASH=0.009489` for each period.

### Stochastic Transaction Costs

Canonical schema (required):

- `kind`: Must be `regime_stochastic`.
- `trade_cost_bps`: Required under each regime; defines the basis-point cost distribution.
- Top-level regime blocks: regime names (for example `calm`, `stress`) are direct keys under `costs`, and each regime block must include `trade_cost_bps`.
- For `kind: lognormal`, `mean` is the target arithmetic mean in basis points. The runner converts it to NumPy's log-space `mu` internally before sampling. Use `mu` or `log_mean` only for explicit log-space configuration.

Concrete canonical example:

```yaml
costs:
  kind: regime_stochastic
  default_regime: calm
  calm:
    trade_cost_bps:
      kind: lognormal
      mean: 6      # arithmetic mean all-in cost, in bps
      sigma: 0.25  # log-space volatility
  stress:
    trade_cost_bps:
      kind: lognormal
      mean: 18     # arithmetic mean all-in cost, in bps
      sigma: 0.35  # log-space volatility
    slippage_multiplier: 1.5  # Optional
```

**Per rebalance date, the cost process outputs:**
- Expected cost in basis points
- Optional slippage multiplier
- Applied to portfolio turnover

## Backward Compatibility

The parser in `src/trend_analysis/monte_carlo/costs.py` still accepts these legacy
keys and aliases. New scenario files and documentation examples should use the
canonical direct-regime shape above.

| Legacy key or alias | Status | Canonical equivalent |
|---------------------|--------|----------------------|
| `costs.regimes.<name>` | `supported` | `costs.<name>` |
| `costs.regimes.<name>.distribution` | `supported` | `costs.<name>.trade_cost_bps` |
| `costs.<name>.distribution` | `supported` | `costs.<name>.trade_cost_bps` |
| `costs.<name>.trade_cost_bps.dist` | `supported` | `costs.<name>.trade_cost_bps.kind` |
| `costs.<name>.trade_cost_bps.std` | `supported` | `costs.<name>.trade_cost_bps.sigma` for lognormal costs, or `std` for normal costs |
| `costs.regimes.<name>: <number>` | `supported` | `costs.<name>.trade_cost_bps.kind: fixed` plus `value: <number>` |
| `costs.default` | `deprecated` | Use `costs.default_regime` for the selected regime label and an explicit direct-regime block for its cost distribution |

Mapping examples:

| Legacy shape | Canonical shape |
|--------------|-----------------|
| `costs.regimes.calm.distribution.kind: fixed` | `costs.calm.trade_cost_bps.kind: fixed` |
| `costs.regimes.calm.distribution.value: 6` | `costs.calm.trade_cost_bps.value: 6` |
| `costs.regimes.stress.slippage_multiplier: 1.5` | `costs.stress.slippage_multiplier: 1.5` |
| `costs.regimes.calm: 6` | `costs.calm.trade_cost_bps.kind: fixed` and `costs.calm.trade_cost_bps.value: 6` |

### Turnover Constraints Under MC

Turnover cap can be:

| Type | Description |
|------|-------------|
| Fixed | Same cap for all paths and periods |
| Sampled | Drawn per strategy at scenario start |
| Regime-Conditional | Tighter in stress periods |

```yaml
# Regime-conditional example
portfolio:
  max_turnover:
    calm: 0.15
    stress: 0.08
```

---

## Multi-Period Integration

### What Fits Well

The existing multi-period infrastructure is already the right abstraction:
- Multi-period schedules define rebalance dates
- Modular selection/weighting/rebalancing via config
- Score frame computation per window

### What Needs Adaptation

**Problem:** Running N strategies independently computes expensive metrics N times.

**Solution:** Superset metrics computation

1. **At scenario start:** Compute union of all metric columns across all strategies
2. **Per path:** Compute those metrics once per rebalance window → `scores_by_date[date] = frame`
3. **Per strategy:** Select relevant columns from cached frame → run lightweight evaluation

```python
# Pseudocode
all_metrics = union(strategy.required_metrics for strategy in strategies)

for path_id in range(n_paths):
    prices = model.sample_prices(...)
    
    # EXPENSIVE: Done once per path
    scores_by_date = {}
    for rebal_date in schedule:
        window = get_in_sample_window(prices, rebal_date)
        scores_by_date[rebal_date] = compute_metrics(window, all_metrics)
    
    # CHEAP: Done per strategy
    for strategy in strategies:
        result = evaluate_strategy(strategy, prices, scores_by_date)
        store_result(path_id, strategy, result)
```

---

## Fold Support

### Purpose

Folds (vintages) test robustness across different calibration/forecast periods.

### Implementation

```yaml
folds:
  enabled: true
  mode: explicit_dates  # Or: rolling, count_spaced
  fold_starts: ["2010-01", "2012-01", "2014-01", "2016-01", "2018-01"]
  calibration_lookback_years: 10
```

**Each fold changes:**
- Calibration window end date (where return model is fit)
- Forecast start date
- Optionally: available universe (for birth/death realism)

### Recommended Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| Number of folds | 3–5 | Balance robustness vs compute |
| Paths per fold | 300–800 | Reduce per-fold to stay in budget |
| Strategy set | Same across folds | Enable fair comparison |

### Output Options

1. **Fold-separated:** Report distributions per fold
2. **Pooled:** Combine across folds (clearly labeled)

---

## Scenario Library

### File Layout

```
config/scenarios/monte_carlo/
├── index.yml                    # Scenario registry
├── hf_equity_ls_10y.yml        # Example scenario
├── hf_macro_20y.yml
└── strategies/
    ├── equity_ls_curated.yml   # Reusable strategy packs
    └── macro_curated.yml
```

### Registry Format (index.yml)

```yaml
scenarios:
  - name: hf_equity_ls_10y
    path: hf_equity_ls_10y.yml
    description: "Equity L/S hedge fund sleeve 10-year forecast"
    tags: [equity, hedge_fund, production]
    
  - name: hf_macro_20y
    path: hf_macro_20y.yml
    description: "Global macro sleeve 20-year stress test"
    tags: [macro, hedge_fund, stress_test]
```

### Complete Scenario Schema

```yaml
scenario:
  name: hf_equity_ls_10y
  description: "Equity L/S hedge fund sleeve 10-year forecast"
  version: "1.0"
  folds:
    enabled: false
    fold_starts: []
    calibration_lookback_years: 10
  return_model:
    kind: stationary_bootstrap  # stationary_bootstrap | regime_conditioned_bootstrap
    mean_block_len: 6
    regime:
      enabled: true
      kind: proxy_vol_threshold
      proxy_column: SPX
      threshold_percentile: 75
  costs:
    kind: regime_stochastic
    default_regime: calm
    calm:
      trade_cost_bps:
        kind: lognormal
        mean: 6      # arithmetic mean all-in cost, in bps
        sigma: 0.25  # log-space volatility
    stress:
      trade_cost_bps:
        kind: lognormal
        mean: 18     # arithmetic mean all-in cost, in bps
        sigma: 0.35  # log-space volatility
      slippage_multiplier: 1.5

# Reference to base pipeline config
base_config: config/defaults.yml

# Monte Carlo settings
monte_carlo:
  mode: two_layer              # two_layer | mixture
  n_paths: 2000
  horizon_years: 10
  frequency: M                  # M | D
  seed: 12345
  jobs: 8                       # Parallel workers

# Strategy configurations
strategy_set:
  curated:
    - name: Rank_12_Equal_TightTurnover
      overrides:
        portfolio:
          selection_mode: rank
          rank:
            n: 12
            metric: Sharpe
          weighting_scheme: equal
          max_turnover: 0.12

    - name: ThresholdHold_Bayes_ModerateTurnover
      overrides:
        portfolio:
          selection_mode: rank
          rank:
            inclusion_approach: threshold
            threshold: 0.3
          weighting_scheme: shrinkage
          max_turnover: 0.18

  sampled:
    enabled: true
    n_strategies: 100
    sampling:
      portfolio.rank.n:
        dist: categorical
        values: [8, 12, 16]
      portfolio.max_turnover:
        dist: uniform
        low: 0.08
        high: 0.25

# Output configuration
outputs:
  directory: outputs/monte_carlo/{scenario_name}/{timestamp}
  store_paths:
    nav: 50                     # Store NAV series for N representative paths
    weights: 0                  # Store weight series (0 = none)
  quantiles: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
  formats: [parquet, csv]       # Output formats
```

`strategy_set` and `outputs` are optional top-level sections. `costs`, `folds`, and
`return_model` may be defined inside `scenario` as shown; `costs` may also be
defined as a top-level compatibility section. Optional sections may be omitted or
set to `null` to document intentional absence where allowed. Required sections such
as `scenario`, `base_config`, and `monte_carlo` remain strict and must contain
valid values (`monte_carlo: null` fails validation).

---

## Output Specifications

### Output Bundle Structure

`mc run` uses a flat output bundle. The CLI export writes `results.<fmt>` and
`summary.<fmt>` into the bundle root, and `manifest.json` records that exact CLI
file map under `outputs.files`. Scenario-configured runner exports may also
write optional diagnostics, pooled/cross-fold summaries, aggregation files, and
NAV-path artifacts into a root-level bundle. The command does not write a
separate frozen scenario YAML file; the manifest records scenario metadata,
runtime overrides, data path, settings, row counts, and the CLI-exported
filenames.

```
outputs/monte_carlo/hf_equity_ls_10y/2026-01-03_143052/
├── manifest.json              # Run metadata + file index
├── results.csv                # Per-path, per-strategy simulation results
├── summary.csv                # Strategy summary statistics
├── diagnostics.csv            # Optional diagnostics table
├── cross_fold_summary.csv     # Optional cross-fold summary table
├── pooled_summary.csv         # Optional pooled summary table
├── pooled_distributions.csv   # Optional pooled distribution table
├── nav_paths.parquet          # Optional root-level NAV paths for mc viz
├── nav_paths_fold_1.parquet   # Optional fold-specific NAV paths
├── path_summary.csv           # Aggregation path summary
├── per_strategy_stats.csv     # Backward-compatible path summary alias
├── per_strategy_path.csv      # Backward-compatible path summary alias
├── quantiles.csv              # Distribution quantiles
├── summary_quantiles.csv      # Human-readable quantile summary
├── breach_probabilities.csv   # Drawdown/metric breach probabilities
└── expected_shortfall.csv     # Tail-risk expected shortfall
```

### Key Output Tables

**summary_quantiles.csv:**
```csv
strategy,metric,q01,q05,q10,q25,q50,q75,q90,q95,q99
Rank_12_Equal,sharpe,0.12,0.28,0.35,0.48,0.62,0.78,0.91,1.02,1.18
Rank_12_Equal,max_dd,-0.52,-0.41,-0.35,-0.28,-0.21,-0.15,-0.11,-0.09,-0.06
...
```

**per_strategy_stats.parquet:**
- Columns: `strategy`, `path_id`, `fold_id`, `sharpe`, `cagr`, `vol`, `max_dd`, `time_underwater`, `terminal_wealth`, `total_turnover`, `total_costs`
- One row per (strategy, path, fold)

### Metrics Computed

| Metric | Description |
|--------|-------------|
| Sharpe | Annualized risk-adjusted return |
| CAGR | Compound annual growth rate |
| Volatility | Annualized standard deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Time Underwater | Periods below previous high-water mark |
| Terminal Wealth | Final portfolio value (starting at 1.0) |
| Expected Shortfall | Average of worst X% outcomes |
| Drawdown Breach Prob | P(max_dd < threshold) |

---

## CLI and Streamlit Integration

### CLI Commands

```bash
# List available scenarios
trend mc list
trend mc list --tags hedge_fund

# Validate scenario configuration
trend mc validate config/scenarios/monte_carlo/hf_equity_ls_10y.yml

# Run scenario
trend mc run --scenario hf_equity_ls_10y --out outputs/mc_run_1
trend mc run --scenario hf_equity_ls_10y --n-paths 500 --jobs 4

# Quick test run
trend mc run --scenario hf_equity_ls_10y --dry-run --n-paths 10

# Cost regime example dry-run (exact command)
trend mc run --scenario cost_regime_example --dry-run --n-paths 10

# Export MC charts from an existing bundle
trend mc viz \
  --bundle outputs/mc_run_1 \
  --out outputs/mc_run_1_exports \
  --charts fan,path_dist,risk_return \
  --html --json --png
```

### Streamlit UI

New tab: **Monte Carlo Simulation**

**Components:**
1. **Scenario Picker** - Dropdown populated from registry
2. **Runtime Overrides** - Adjust n_paths, horizon, seed, jobs
3. **Run Button** - Execute with progress bar
4. **Distribution Plots** - Histograms, fan charts, quantile plots
5. **Summary Tables** - Strategy comparison grid
6. **Export Links** - Download output bundle

**Progress Feedback:**
- Path completion counter
- Estimated time remaining
- Strategy evaluation progress

---

## Implementation Milestones

### Milestone 1: Scenario + Config Foundations
- [x] Monte Carlo scenario schema + loader (`src/trend_analysis/monte_carlo/scenario.py`)
- [x] Scenario library registry + discovery (`src/trend_analysis/monte_carlo/registry.py`)

### Milestone 2: Return Models
- [x] Price-path model interface + utilities (`src/trend_analysis/monte_carlo/models/base.py`)
- [x] Multivariate stationary bootstrap implementation (`src/trend_analysis/monte_carlo/models/bootstrap.py`)
- [x] Regime labeling + regime-conditioned bootstrap (`src/trend_analysis/monte_carlo/models/regime.py`)

### Milestone 3: Strategy Sets
- [x] Strategy variant representation + config merge (`src/trend_analysis/monte_carlo/strategy/variant.py`)
- [x] Sampled strategy generator (`src/trend_analysis/monte_carlo/strategy/sampler.py`)
- [x] Initial curated strategy packs (`config/scenarios/monte_carlo/strategies/`)

### Milestone 4: MC Runner + Caching
- [x] MonteCarloRunner (two-layer + mixture modes) (`src/trend_analysis/monte_carlo/runner.py`)
- [x] Path-context caching (score frames) (`src/trend_analysis/monte_carlo/runner.py`)
- [x] Common random numbers support (`src/trend_analysis/monte_carlo/runner.py`)

### Milestone 5: Costs, Turnover, Cash
- [x] Explicit cash handling + RF accrual (`src/trend_analysis/monte_carlo/runner.py`)
- [x] Regime-dependent stochastic costs (`src/trend_analysis/monte_carlo/costs.py`)
- [x] Turnover constraint variation under MC (`src/trend_analysis/monte_carlo/runner.py`)

### Milestone 6: Folds + UX
- [x] Fold/vintage support (`src/trend_analysis/monte_carlo/folds.py`)
- [x] Aggregation outputs + distributions (`src/trend_analysis/monte_carlo/aggregator.py`)
- [x] CLI commands (`src/trend_analysis/cli.py`)
- [x] Streamlit MC tab (`streamlit_app/pages/5_Monte_Carlo.py`)

---

## Compute Planning

### Typical Run Scale

| Parameter | Typical Value |
|-----------|---------------|
| Paths | 500–5,000 |
| Curated Strategies | 12–30 |
| Sampled Variants | 50–200 |
| Total Evaluations | 2,000 × 20 = 40,000 |

### Performance Requirements

For feasible runtimes:
1. Path context computed **once** per path
2. Strategy evaluation is **lightweight** (uses cached score frames)
3. Parallelization over paths (configurable `jobs` parameter)

### Estimated Runtime (8-core machine)

| Paths | Strategies | Est. Runtime |
|-------|------------|--------------|
| 500 | 20 | ~5 minutes |
| 2,000 | 20 | ~20 minutes |
| 2,000 | 100 | ~45 minutes |
| 5,000 | 100 | ~2 hours |

*Note: Estimates assume efficient caching. Without path-context caching, multiply by strategy count.*

---

## Related Documentation

- [docs/phase-2/Agents.md](../phase-2/Agents.md) - Phase 2 implementation spec
- [Agents.md](../../Agents.md) - Project guidance and guard-rails
- [config/defaults.yml](../../config/defaults.yml) - Base pipeline configuration

---

## Appendix: Issue Tracking

See GitHub Issues labeled `phase:monte-carlo` for implementation tickets.

Epic: [Monte Carlo Framework](#) (link to be added after issue creation)
