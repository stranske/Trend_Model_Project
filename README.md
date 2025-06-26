# Trend Model Project

This repository contains experiments and utilities for analyzing volatility-adjusted trend portfolios. The Jupyter notebooks demonstrate how to load hedge fund data, apply trend-following rules, and export the results.

## Notebooks

- `Vol_Adj_Trend_Analysis1.2.TrEx.ipynb` – an earlier version of the analysis.
- `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` – the current main notebook showing the full workflow.
- Additional historical notebooks can be found under `notebooks/old` and `Old/`.

## Setup

1. Create a virtual environment and install the required packages:
   ```bash
   ./scripts/setup_env.sh
   ```
   This bootstraps a `.venv` directory and installs everything from
   `requirements.txt`, which includes `pandas`, `numpy`, `matplotlib`,
   `ipywidgets` and `xlsxwriter`.
2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` and run the cells in order.

## Applying patches

Some updates are provided as patch files. Apply them from the repository root with:

```bash
git apply -p1 <patchfile>
```

Replace `<patchfile>` with the patch you want to apply (for example `codex.patch`).

## Command-line usage

You can also run the analysis pipeline directly from the command line. Invoke
the entry point with an optional configuration file:

```bash
python -m trend_analysis.run_analysis -c path/to/config.yml
```

The configuration file **must** define `data.csv_path` pointing to your CSV
data. If ``-c`` is omitted, ``run_analysis`` loads
`config/defaults.yml`, or the path set via the ``TREND_CFG`` environment
variable:

```bash
TREND_CFG=custom.yml python -m trend_analysis.run_analysis
```


## Ranking-based selection

`portfolio.selection_mode` supports a new `rank` value for picking funds by
performance metrics. The defaults for this mode live under `portfolio.rank` in
`config/defaults.yml`. Metrics can be combined using z-scored weights so they
are comparable across scales.

## Testing

Install the project dependencies before running the test suite. This can be done using the setup script, which
creates a virtual environment and installs everything from `requirements.txt`
(including `pytest`):

```bash
./scripts/setup_env.sh
```

or by installing directly with pip (Python 3.11+ recommended):

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, run the tests with coverage enabled:

```bash
pytest --cov trend_analysis --cov-branch
```

Alternatively, you can use the helper script which installs the requirements
and then executes the test suite in one step:

```bash
./scripts/run_tests.sh
```
