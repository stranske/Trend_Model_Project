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
| `score_prop_bayes`                   | `ScorePropBayesian`        | Shrunk scores: score_shrunk_i = μ + (1-τ)(score_i-μ)      | Dampens noise / overfitting                     | Choose τ (`shrink_tau`) sensibly (0–1)       | `method: score_prop_bayes\n    params:\n      shrink_tau: 0.25` |
| `adaptive_bayes`                     | `AdaptiveBayesWeighting`   | Online Bayesian update of score distribution              | Learns over periods; path-aware                | Needs state persistence for best effect      | `method: adaptive_bayes` |
| `risk_parity`                        | `RiskParity`               | Target equal marginal risk contribution                   | Balances vol; diversification                  | Can overweight low‑vol/low‑return assets     | `method: risk_parity` |
| `hrp`                                | `HierarchicalRiskParity`   | Tree clustering → recursive bisection risk allocation     | Robust to collinearity; hierarchical structure | Slightly heavier compute; cluster instability| `method: hrp` |
| `erc`                                | `EqualRiskContribution`    | Solve for RC_i = TotalRisk / N                            | Formal equal risk targeting                    | Optimisation may fail on degenerate matrices | `method: erc` |

Notes:
1. All engines normalise final weights to sum to 1.0 (100%).
2. Negative intermediate scores are truncated to zero for score‑proportional engines before normalisation.
3. Bayesian shrinkage parameter `shrink_tau` closer to 1.0 = stronger pull toward cross‑sectional mean.
4. If an engine raises a recoverable error (e.g. singular covariance) the pipeline logs a single WARNING and falls back to `EqualWeight`, recording structured `fallback_info` visible in the CLI and GUI.
5. Additional third‑party / custom engines can be registered via the weighting plugin registry (see `docs/plugin-interface.md`).

## 7. Output formats

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

## 14. Further help

See `README.md` for a short overview of the repository structure and the example notebooks for end‑to‑end demonstrations.

## 15. Turnover and Transaction Cost Controls

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

