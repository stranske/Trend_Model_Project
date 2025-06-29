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
 This helper script lives in `scripts/setup_env.sh` and simply wraps
 `python -m venv` followed by `pip install -r requirements.txt`.  It
 creates a `.venv` directory and installs everything from
 `requirements.txt`, including `pandas`, `numpy`, `matplotlib`,
 `ipywidgets`, `PyYAML` and `xlsxwriter`.
2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` and run the cells in
   order.  All notebooks reside at the repository root and depend on the
   `trend_analysis` package for data loading, metrics and exports.

## Applying patches

Some updates are provided as patch files. Apply them from the repository root with:

```bash
git apply -p1 <patchfile>
```
The patches usually update modules under the `trend_analysis/` package,
allowing you to rebuild the library incrementally.

Replace `<patchfile>` with the patch you want to apply (for example `codex.patch`).

## Command-line usage

You can also run the analysis pipeline directly from the command line. Invoke
the entry point with an optional configuration file:

```bash
python -m trend_analysis.run_analysis -c path/to/config.yml
```
This command invokes `main()` in `trend_analysis/run_analysis.py`.  That
script loads the configuration via `trend_analysis.config.load()` and
then runs the pipeline defined in `trend_analysis/pipeline.py`.

The configuration file **must** define `data.csv_path` pointing to your CSV
data. If ``-c`` is omitted, ``run_analysis`` loads
`config/defaults.yml`, or the path set via the ``TREND_CFG`` environment
variable:

```bash
TREND_CFG=custom.yml python -m trend_analysis.run_analysis
```
Here the environment variable ``TREND_CFG`` points the loader in
``trend_analysis.config`` to your custom YAML file, ensuring the same
``main()`` function from `run_analysis.py` uses your overrides.


## Ranking-based selection

`portfolio.selection_mode` supports a new `rank` value for picking funds by
performance metrics. The defaults for this mode live under `portfolio.rank` in
`config/defaults.yml`. Metrics can be combined using z-scored weights so they
are comparable across scales.
The actual ranking logic is implemented in
`trend_analysis/core/rank_selection.py` and wired into the pipeline via
`trend_analysis/pipeline.py`.

## Testing

Install the project dependencies (such as `pandas`, `numpy` and `PyYAML`) before running the test suite. This can be done using the setup script, which
creates a virtual environment and installs everything from `requirements.txt`
(including `pytest`):

```bash
./scripts/setup_env.sh
```
The script in `scripts/setup_env.sh` prepares a throw-away environment
and installs test dependencies such as `pytest`.

or by installing directly with pip (Python 3.11+ recommended):

```bash
pip install -r requirements.txt
```
`requirements.txt` lives at the repository root and lists every
dependency required to run the analysis and its tests.

Once the dependencies are installed, run the tests with coverage enabled:

```bash
pytest --cov trend_analysis --cov-branch
```
All unit tests reside in the `tests/` directory and enforce 100 % branch
coverage through `pytest-cov`.

Alternatively, you can use the helper script which installs the requirements
and then executes the test suite in one step:

```bash
./scripts/run_tests.sh
```
This convenience wrapper (under `scripts/run_tests.sh`) installs the
requirements and then runs the same `pytest` command as above.
