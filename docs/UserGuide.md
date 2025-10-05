# User Guide

This document introduces the main features of the Trend Model Project and explains how to use them. It assumes no prior knowledge of the system.

## 1. Setup

1. Install the dependencies into a virtual environment:
   ```bash
   ./scripts/setup_env.sh
   ```
   This creates `.venv/` and installs packages from `requirements.txt`.
2. Optional: run the test suite to verify everything works:
   ```bash
   ./scripts/run_tests.sh
   ```

## 2. Running the analysis from the command line

Invoke the pipeline with an optional YAML configuration file:
```bash
python -m trend_analysis.run_analysis -c path/to/config.yml
```
If `-c` is omitted, the tool loads `config/defaults.yml` or the file specified by the `TREND_CFG` environment variable.

The metrics output is printed to the console and can also be written to Excel, CSV or JSON depending on `output.format` in the config.

### 2.1 Input frequency and missing-data handling

- The ingestion layer inspects the `Date` column and automatically classifies the cadence as daily, weekly or monthly. Holiday gaps and February shortfalls are tolerated and the series is resampled to month-end returns before modelling begins.
- Missing-data behaviour is configured per asset via `data.missing_policy` (`drop`, `ffill` or `zero`). When forward-fill is selected the optional `data.missing_fill_limit` caps the length of consecutive gaps that will be filled.
- Every report—console text, Excel summary and the JSON bundle—prints a one-line status showing the detected frequency and the applied missing-data policy so runs remain auditable.

## 3. Interactive GUI

A graphical interface is available in Jupyter. Launch it by running:
```python
from trend_analysis.gui import launch
launch()
```

The GUI allows you to edit the configuration, select the portfolio mode, tweak ranking options and choose a weighting method. A dark‑mode toggle controls the theme. All settings persist to `~/.trend_gui_state.yml` so the next session picks up where you left off.

After editing the parameters click **Run** to execute the pipeline. Results are exported according to the chosen format. Weighting plug‑ins discovered via entry points appear automatically in the interface.

## 4. Configuration presets

Ready-made presets live in `config/presets/` and provide sensible defaults for common risk profiles.
Select a preset on the Streamlit **Configure** page or load it from the command line:

```bash
python -m trend_analysis.run_analysis -c config/presets/balanced.yml
```

Replace `balanced` with `conservative` or `aggressive` as needed. See
[PresetStrategies.md](PresetStrategies.md) for a summary of each option.

The CLI also supports merging signal presets into any configuration without
swapping YAML files. Supply `--preset` to `trend-model run` and the TrendSpec
parameters (window, lag, volatility scaling, and z-score toggle) are injected
before the analysis starts:

```bash
trend-model run -c config/demo.yml -i data/returns.csv --preset Aggressive
```

On the Streamlit side the **Trend Signal Settings** card mirrors the registry.
Selecting “Conservative” or “Aggressive” updates the window, minimum period,
volatility scaling and target sliders immediately so the form and CLI stay in
sync.

### 4.1 Handling missing data

Two configuration knobs control how sparse series are treated during ingest:

| Setting | Type | Effect |
|---------|------|--------|
| `data.missing_policy` | string _or_ mapping | Choose `drop`, `ffill`, or `zero`. A mapping supports per-column overrides (use `"*"` for the default). |
| `data.missing_limit` | integer _or_ mapping | Maximum length of consecutive gaps (in periods) that may be filled per column. `null` means unlimited. |

Examples:

```yaml
data:
   missing_policy: "ffill"         # forward-fill short gaps everywhere
   missing_limit: 2                 # tolerate up to two consecutive missing months

   # Override: drop FundZ instead of filling
   missing_policy:
      "*": "ffill"
      FundZ: "drop"
   missing_limit:
      "*": 2
      FundZ: 0
```

The validator records the applied policy in the metadata (including dropped columns and fill counts) and surfaces the summary in CLI and Streamlit reports. Frequency detection also respects the configured limits, allowing—for example—multi-week holiday gaps when `missing_limit` is large enough.

## 5. Selection modes and ranking

`portfolio.selection_mode` supports `all`, `random`, `manual` and `rank` values. In rank mode you can keep the top funds by score or apply a threshold. Scores come from metrics defined under `metrics.registry` and may be combined with z‑scored weights.

Manual mode displays a table of candidates so you can override fund inclusion and weights directly.

## 6. Weighting options

Several weighting strategies are built in:

- **EqualWeight** – simple 1/N allocation.
- **ScorePropSimple** – weights proportional to positive scores.
- **ScorePropBayesian** – shrinks scores toward the mean before weighting.
- **AdaptiveBayesWeighting** – updates weights over multiple periods and persists its state between runs.

When the GUI is used, the state of AdaptiveBayesWeighting is saved alongside the main config so that repeated runs refine the posterior.

### 6.1 Weighting engine reference

| Engine (`portfolio.weighting.method`) | Class / Plugin Name        | Core Idea / Formula (informal)                            | Strengths                                      | Caveats / When to Avoid                     | Minimal YAML Snippet |
|--------------------------------------|----------------------------|-----------------------------------------------------------|------------------------------------------------|----------------------------------------------|----------------------|
| `equal`                              | `EqualWeight`              | w_i = 1 / N                                               | Stable, transparent, zero parameter risk       | Ignores differences in risk / edge           | `portfolio:\n  weighting:\n    method: equal` |
| `score_prop`                         | `ScorePropSimple`          | w_i ∝ max(score_i, 0); normalised                         | Directly reflects relative strength            | Sensitive to extreme outliers               | `method: score_prop` |
| `score_prop_bayes`                   | `ScorePropBayesian`        | Shrunk scores: score'_i = μ + (1-τ)(score_i-μ)            | Dampens noise / overfitting                     | Choose τ (`shrink_tau`) sensibly (0–1)       | `method: score_prop_bayes\n    params:\n      shrink_tau: 0.25` |
| `adaptive_bayes`                     | `AdaptiveBayesWeighting`   | Online Bayesian update of score distribution              | Learns over periods; path‑aware                | Needs state persistence for best effect      | `method: adaptive_bayes` |
| `risk_parity`                        | `RiskParity`               | Target equal marginal risk contribution                   | Balances vol; diversification                  | Can overweight low‑vol/low‑return assets     | `method: risk_parity` |
| `hrp`                                | `HierarchicalRiskParity`   | Tree clustering → recursive bisection risk allocation     | Robust to collinearity; hierarchical structure | Slightly heavier compute; cluster instability| `method: hrp` |
| `erc`                                | `EqualRiskContribution`    | Solve for RC_i = TotalRisk / N                            | Formal equal risk targeting                    | Optimisation may fail on degenerate matrices | `method: erc` |

Notes:
1. All engines normalise final weights to sum to 1.0 (100%).
2. Negative intermediate scores are truncated to zero for score‑proportional engines before normalisation.
3. Bayesian shrinkage parameter `shrink_tau` closer to 1.0 = stronger pull toward cross‑sectional mean.
4. If an engine raises a recoverable error (e.g. singular covariance) the pipeline logs a single WARNING and falls back to `EqualWeight`, recording structured `fallback_info` visible in the CLI and GUI.
5. Additional third‑party / custom engines can be registered via the weighting plugin registry (see `docs/plugin-interface.md`).

## 7. Risk controls

Volatility targeting and basic constraints live under the `vol_adjust` and
`portfolio` sections of the configuration. The key knobs are:

| Setting | Type | Effect |
|---------|------|--------|
| `vol_adjust.target_vol` | float | Annualised volatility target applied to the portfolio. |
| `vol_adjust.window.length` | integer | Lookback window (in periods) for realised-vol calculations. |
| `vol_adjust.window.decay` | string | `simple` for rolling window or `ewma` for exponentially weighted volatility. |
| `vol_adjust.window.lambda` | float | EWMA decay factor when `decay: ewma` (0 < λ < 1). |
| `vol_adjust.floor_vol` | float | Minimum annualised volatility per asset to avoid excessive leverage. |
| `vol_adjust.warmup_periods` | integer | Number of initial rows with zero exposure after re-scaling. |
| `portfolio.max_turnover` | float | Turnover cap (fraction of the book) enforced at each rebalance. |
| `portfolio.constraints.long_only` | bool | Clip negative weights before normalisation. |
| `portfolio.constraints.max_weight` | float | Maximum weight per asset before normalisation. |

The CLI summary and Streamlit app now display a “Risk diagnostics” panel that
highlights the most recent realised volatility per asset, the portfolio
volatility, and the turnover applied at the latest rebalance. These diagnostics
are also included in the JSON bundle and the formatted text summary generated
by the exporter.

## 8. Output formats

Set `output.format` to `excel`, `csv` or `json`. Excel exports include a formatted summary sheet generated by `trend_analysis.export.make_summary_formatter` and can be inspected with any spreadsheet program.

## 8. Benchmarks and information ratio

Add a `benchmarks` mapping in the config to compute information ratios against one or more indices. The results appear as `ir_<label>` columns in the metrics output and on the summary sheet.

## 9. Inspecting the score frame

`run_analysis()` returns a dictionary containing a `score_frame` DataFrame with one column per metric and attributes describing the analysed period. Advanced users can examine this table to build custom selection logic.

## 10. Multi-period demo

Generate the synthetic dataset and run the helper script to exercise the Phase 2 engine. The demo now starts by bootstrapping a clean environment and then cycles through several selector and weighting strategies while exporting results in multiple formats. `config/demo.yml` enables the rank-based selector and registers an `SPX` benchmark so information ratios are computed:

```bash
./scripts/setup_env.sh
python scripts/generate_demo.py [--no-xlsx]
python scripts/run_multi_demo.py
```
Use `--no-xlsx` to skip generating the Excel workbook if binary files should be
avoided.

`run_multi_demo.py` calls ``export.export_data`` so CSV, Excel, JSON **and TXT** reports are produced in one go. It also runs the full test suite, ensuring multiple periods are processed and that adaptive weights evolve over time.
The script exercises the CLI wrappers as part of these checks.

## 11. Demo pipeline (maintenance / CI)

Whenever the exporter or pipeline behaviour changes, re-run the demo steps below
and adjust `config/demo.yml` or `scripts/run_multi_demo.py` so the demo
exercises every new code path.

1. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
2. **Generate the demo dataset**
   ```bash
   python scripts/generate_demo.py
   ```
3. **Run the full demo pipeline and export checks**
   ```bash
   python scripts/run_multi_demo.py
   ```
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```

## 12. Reproducibility and Testing

For reproducible results across different runs and environments, see [ReproducibilityGuide.md](ReproducibilityGuide.md). This is especially important when using random selection modes or when running tests.

## 12.1 Walk-forward (rolling OOS) analysis

The Streamlit Results page now includes a Walk-forward analysis expander to aggregate metrics over rolling out-of-sample windows and, optionally, by regime. See [Walkforward.md](Walkforward.md) for a quick guide and a CLI example.

## 13. Structured run logging

Structured JSONL logging provides a machine‑parsable trace of each pipeline run. It is enabled by default for CLI and Streamlit executions.

Key points:

- Default path: `outputs/logs/run_<run_id>.jsonl` (auto‑created). A unique 12‑char hex `run_id` is generated if the config does not supply one.
- CLI overrides: `--log-file custom/path.jsonl` writes to a specific file; `--no-structured-log` disables JSONL emission entirely.
- Rotation: log files rotate when they exceed ~1 MB (single backup `<name>.jsonl.1`).
- Schema (one JSON object per line): `{"ts": <epoch_seconds>, "run_id": "...", "step": "selection", "level": "INFO", "msg": "...", "extra": {...}}`
- Steps include: init / start, load_data, metrics_build, selection, weighting (always present via fallback), benchmarks, pipeline_complete, summary_render, export_start / export_complete, bundle_complete, end.
- Streamlit UI: the Results page shows a Run Log pane with auto‑refresh (5s), tail view, file size & line count, error summary and a download button.

Command examples:

```bash
# Default (creates outputs/logs/run_<id>.jsonl)
python -m trend_analysis.run_analysis -c config/demo.yml -i demo/demo_returns.csv

# Custom path
python -m trend_analysis.run_analysis -c config/demo.yml -i demo/demo_returns.csv \
   --log-file logs/my_run.jsonl

# Disable structured logging
python -m trend_analysis.run_analysis -c config/demo.yml -i demo/demo_returns.csv \
   --no-structured-log
```

Programmatic helpers:

- `from trend_analysis.logging import logfile_to_frame, error_summary`
- `logfile_to_frame(path)` returns a DataFrame (most recent first) with flattened scalar fields from `extra`.
- `error_summary(path)` groups ERROR lines with counts and last timestamp.

Notebook inspection example:

```python
from pathlib import Path
from trend_analysis.logging import logfile_to_frame, error_summary
log_path = next(Path('outputs/logs').glob('run_*.jsonl'))
df = logfile_to_frame(log_path)
print(df.head())
print(error_summary(log_path))
```

Disable logging only if you have strict I/O limits or are micro‑benchmarking; overhead is negligible (dozens of lines per run).

## 14. Regime tagging and reporting

The default configuration now ships with a `regime` block that tags each
out-of-sample period as **Risk-On** or **Risk-Off**. By default the engine
compounds six months of index returns, optionally smooths the signal, and
compares the result with the configured threshold. Positive or neutral readings
map to Risk-On while negative readings map to Risk-Off. Switching `method` to
`volatility` instead evaluates a rolling realised-volatility filter: returns are
converted to standard deviation (optionally annualised) and periods below the
threshold are tagged as Risk-On. All controls can be tuned without touching the
code:

```yaml
regime:
  enabled: true          # disable to skip regime analysis entirely
  proxy: "SPX"           # column in the returns frame or uploaded index file
  method: "rolling_return"
  lookback: 126          # number of observations to compound
  smoothing: 3           # optional moving average over the rolling return
  threshold: 0.0         # shift the cut-over between regimes
  neutral_band: 0.001    # treat small deviations as neutral noise
  min_observations: 4    # minimum rows required to compute metrics
  annualise_volatility: true  # only used when method: volatility
```

When `regime.enabled` is true the CLI summary, Excel workbook, JSON/CSV/TXT
exports, and the unified HTML/PDF report include a **Performance by Regime**
table. It lists CAGR, Sharpe, max drawdown, hit rate, and the observation count
for the user-weight portfolio (and equal-weight baseline when available) across
Risk-On, Risk-Off and aggregate windows. Any regime with fewer than
`min_observations` samples is shown as `N/A` and annotated with a descriptive
footnote. The unified report also appends a sentence to the executive summary
and narrative highlighting the relative performance between regimes. When the
strategy excels during Risk-Off periods, the copy explicitly calls out the
outperformance; if both environments trade in line, the message notes that the
regimes behaved similarly.

The `regime_notes` entry in the result dictionary carries the collected
footnotes; they are exported as a one-column table for easy auditing alongside
the numeric breakdown. Supplying your own proxy is as simple as adding the
column to the input data or pointing `regime.proxy` at a custom series in the
indices bundle. When using the volatility method the threshold is interpreted as
annualised volatility when `annualise_volatility: true` (set to `false` to work
with per-period figures).

## 15. Further help

See `README.md` for a short overview of the repository structure and the example notebooks for end‑to‑end demonstrations.

## 16. Turnover and Transaction Cost Controls

Two optional portfolio execution controls make simulation results closer to
realistic implementation:

- `portfolio.transaction_cost_bps` – linear cost, in basis points, applied to
   the absolute turnover each rebalancing period. Must be a non‑negative
   number (e.g. `10` = 10 bps = 0.10%). The summary metrics internally
   subtract these costs when computing risk/return figures.
- `portfolio.max_turnover` – soft cap on total turnover (sum of absolute
   weight changes) for a single rebalance expressed as a fraction of gross
   notional. Accepted range is `0.0` to `2.0` where `1.0` effectively means
   “no practical cap” for most strategies (full liquidation + rebuild would
   require `2.0`). Values above `2.0` are rejected by configuration validation.

Validation rules:

- Negative values for either field raise a configuration error.
- Values are coerced from numeric strings when possible (e.g. `"15"`).
- Omitting both keys preserves previous behaviour (no costs, no cap).

Multi‑period runs attach per‑period `turnover` and `transaction_cost` figures
to each result dictionary. These appear in a separate execution metrics export
(`execution_metrics` sheet / file) so the legacy Phase‑1 summary schema remains
unchanged.

## 17. Portfolio Constraint Set (advanced)

The engine can project preliminary weights onto a feasible region defined by a constraint set:

Supported keys (YAML under `portfolio.constraints`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `long_only` | bool | `true` | Clip negative weights then renormalise. Raises if all become zero. |
| `max_weight` | float | `null` | Per-asset cap (exclusive of redistribution tolerance). Must satisfy `max_weight * N >= 1`. |
| `group_caps` | mapping(str->float) | `null` | Upper bounds for groups; excess is redistributed to other groups. |
| `groups` | mapping(asset->group) | `null` | Required when `group_caps` set; every asset must map to a group present. |
| `cash_weight` | float | `null` | Fixed slice allocated to a synthetic `CASH` line; remaining weights scaled to sum to `1 - cash_weight`. |

Example snippet:

```yaml
portfolio:
   constraints:
      long_only: true
      max_weight: 0.25
      group_caps:
         Trend: 0.55
         Macro: 0.60
      groups:
         FundA: Trend
         FundB: Trend
         FundC: Macro
         FundD: Macro
      cash_weight: 0.10
```

Behaviour notes:
1. Order of enforcement: long_only → normalise → max_weight cap (iterative) → group_caps (with redistribution) → re-apply max_weight if needed → carve out `cash_weight` slice → final normalisation.
2. `cash_weight` feasibility: with N non-cash assets, equal residual weight after carving cash is `(1 - cash_weight)/N`; if this already exceeds `max_weight` a `ConstraintViolation` is raised.
3. `cash_weight` triggers creation of a `CASH` line if not already present; if provided, the existing weight is overwritten by the fixed slice.
4. Group caps whose total (for the groups present) sums to less than 100% lead to an infeasibility error to avoid unallocated mass.
5. All redistribution steps preserve total weight = 1.0 within numerical tolerance.
6. Any infeasibility raises `ConstraintViolation` with a descriptive message (e.g. *"max_weight too small for number of assets"*, *"cash_weight exceeds max_weight constraint"*).

Programmatic usage:

```python
from trend_analysis.engine.optimizer import apply_constraints, ConstraintSet
import pandas as pd

weights = pd.Series({"FundA": 0.4, "FundB": 0.3, "FundC": 0.3})
cs = ConstraintSet(long_only=True, max_weight=0.25, cash_weight=0.1)
projected = apply_constraints(weights, cs)
print(projected)
```

To ignore all constraints (legacy behaviour), omit the `portfolio.constraints` block.

