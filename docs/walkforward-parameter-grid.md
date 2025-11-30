# Walk-Forward Parameter Grid

The `scripts/walk_forward.py` helper runs a deterministic walk-forward sweep
over a small parameter grid so we can assess the stability of demo trend
settings. It consumes a YAML configuration file describing the dataset, window
lengths, and the grid to evaluate.

## Configuration layout

```yaml
data:
  csv_path: ../demo/demo_returns.csv
  date_column: Date
  columns: [Mgr_01, Mgr_02, Mgr_03, Mgr_04]

walk_forward:
  train: 36      # in-sample length (rows)
  test: 12       # out-of-sample rows per fold
  step: 6        # advance the window by 6 rows per split

strategy:
  top_n: 4
  defaults:
    band: 0.0
  grid:
    lookback: [6, 9, 12]
    band: [0.0, 0.0015]

run:
  name: demo-grid
  output_dir: perf/wf
  seed: 42
```

* `lookback` controls how many trailing observations the selector uses when
  ranking managers inside each training fold.
* `band` requires average returns to clear a neutral buffer before they are
  eligible for inclusion. When no manager clears the band the script falls back
  to the best available scores.
* `seed` drives the tie-breaker when multiple managers have identical scores.

## Running the sweep

Execute the CLI with the bundled config:

```bash
python scripts/walk_forward.py --config config/walk_forward.yml
```

Example output:

```
Wrote walk-forward artifacts to /workspace/Trend_Model_Project/perf/wf/demo-grid-20250101-120000
```

The command produces:

* `folds.csv` – per-fold metrics (CAGR, Sharpe, max drawdown, hit rate,
  turnover) together with the selected managers.
* `summary.csv` and `summary.jsonl` – aggregated statistics for each parameter
  combination.
* `mean_cagr_heatmap.png` – optional heatmap when the grid contains exactly two
  varying parameters.
* `config_used.yml` – a copy of the config so runs are reproducible.

Artifacts live under `perf/wf/<run-name>-<timestamp>/` so multiple experiments
can coexist without clobbering prior results.
