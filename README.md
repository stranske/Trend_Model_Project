# Trend Model Project

This repository contains experiments and utilities for analyzing volatility-adjusted trend portfolios. The Jupyter notebooks demonstrate how to load hedge fund data, apply trend-following rules, and export the results.

For a beginner-friendly overview, see [docs/UserGuide.md](docs/UserGuide.md).


## Notebooks

- `Vol_Adj_Trend_Analysis1.2.TrEx.ipynb` – an earlier version of the analysis.
- `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` – the current main notebook showing the full workflow.
- Additional historical notebooks can be found under `notebooks/old` and `Old/`.

## Installation

### From PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install trend-model
```

This provides the ``trend-model`` command with both GUI and pipeline modes:

```bash
trend-model run --help
trend-model gui
```

### Via ``pipx``

For an isolated installation without activating a virtual environment:

```bash
pipx install trend-model
trend-model gui
```

### From Source

For development or to use the latest features:

```bash
git clone https://github.com/stranske/Trend_Model_Project.git
cd Trend_Model_Project
./scripts/setup_env.sh
source .venv/bin/activate
pip install -e .
```

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

The ``trend-model`` command wraps the pipeline and GUI. To run the analysis
from a CSV file and YAML configuration:

```bash
trend-model run -c path/to/config.yml -i returns.csv
```

The configuration file **must** define `data.csv_path`, which is overridden by
the ``-i`` option above. If ``-c`` is omitted, the defaults from
`config/defaults.yml` or the ``TREND_CFG`` environment variable are used.


## Ranking-based selection

`portfolio.selection_mode` supports a new `rank` value for picking funds by
performance metrics. The defaults for this mode live under `portfolio.rank` in
`config/defaults.yml`. Metrics can be combined using z-scored weights so they
are comparable across scales.
The actual ranking logic is implemented in
`trend_analysis/core/rank_selection.py` and wired into the pipeline via
`trend_analysis/pipeline.py`.

## Information ratio & benchmarks

The pipeline also calculates each portfolio's **information ratio** relative to
one or more benchmarks. The YAML configuration accepts a `benchmarks` mapping
of labels to column names:

```yaml
benchmarks:
  spx: SPX
  tsx: TSX
```

When set, additional `OS IR <label>` columns are appended to the metrics output
and summary Excel sheet. These values measure risk‑adjusted performance versus
the chosen benchmarks.

## Score frame

`trend_analysis.pipeline.single_period_run()` returns a **score frame** – a
DataFrame indexed by fund code with one column per metric listed in
`RiskStatsConfig.metrics_to_run`. The frame also carries `insample_len` and
`period` metadata. `run_analysis()` places this table in the result dictionary
under the key `"score_frame"` so callers can inspect the raw metric values
before any ranking or weighting takes place.

Example:
```python
import pandas as pd
from trend_analysis.pipeline import single_period_run, run_analysis

df = pd.read_csv("my_returns.csv")
sf = single_period_run(df, "2021-01", "2021-03")
print(sf)

# Alternatively, call ``run_analysis`` and grab the same table
res = run_analysis(df, "2021-01", "2021-03", "2021-04", "2021-06", 1.0, 0.0)
score_frame = res["score_frame"]
print(score_frame)
```

## Multi-period demo

Phase 2 introduces a multi-period engine with adaptive weighting. A small
demo dataset and helper script are included to exercise this feature. The
`config/demo.yml` file now enables the rank-based selection mode and adds an
`SPX` benchmark so information ratios and multi-format exports are covered.

Generate the synthetic data and run the demo as follows. The helper script now
cycles through several selector and weighting combinations to cover the main
pipeline features:

```bash
./scripts/setup_env.sh

Troubleshooting: If your VS Code terminal exits with code 1 unexpectedly, it may be because a script was sourced that enabled strict bash options (set -e/pipefail) in your interactive shell. Our setup script is now safe to source, but as a rule prefer executing it: `./scripts/setup_env.sh`. If you ever source a script that sets strict modes, start a new terminal to reset the shell state.
python scripts/generate_demo.py [--no-xlsx]
python scripts/run_multi_demo.py
```
Use `--no-xlsx` to skip the Excel workbook when large files are undesired.

The script prints the number of generated periods and verifies that
weights evolve across them, confirming the Phase 2 logic works end to end.
It now exports each period's score frame alongside the single-period metrics
using ``export.export_data`` so CSV, Excel, JSON **and TXT** files are produced in one call.
The demo also exercises the ``all``, ``random`` and ``manual`` selection modes,
and now calls ``single_period_run`` together with ``calc_portfolio_returns`` to validate
the pipeline helpers. The wrapper ``pipeline.run_analysis`` is also invoked, and
extra calls to ``rank_select_funds`` use the ``percentile`` and ``rank`` transforms so all
scoring options are covered. The demo now checks ``quality_filter`` and the dual calling
patterns of ``select_funds``. It verifies AdaptiveBayesWeighting state persistence and runs the CLI via both the ``-c``
flag and the ``TREND_CFG`` environment variable. Additional checks cover edge
cases of ``AdaptiveBayesWeighting`` such as zero half-life behaviour and
invalid ``prior_mean`` lengths. Finally, the script invokes the
full test suite so every module is covered.
It now also calls ``run_multi_analysis.main`` with a temporary config to verify
the dedicated multi-period CLI works and produces CSV output.
The demo also validates the ``Portfolio`` container by rebalancing
weights from both DataFrame and Series inputs so history tracking remains
tested.

## Demo pipeline (maintenance / CI)

Keep the demo scripts in lock‑step with the exporter and pipeline
functionality. Whenever either changes, run the sequence below and update
`config/demo.yml` or `scripts/run_multi_demo.py` so the demo covers every
code path.

See **[docs/DemoMaintenance.md](docs/DemoMaintenance.md)** for a concise
checklist of these steps.

1. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
2. **Generate the demo dataset**
   ```bash
   python scripts/generate_demo.py [--no-xlsx]
   ```
   Pass `--no-xlsx` to avoid creating the Excel copy.
3. **Run the full demo pipeline and export checks**
   ```bash
   python scripts/run_multi_demo.py
   ```
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```

## Interactive GUI

An interactive configuration GUI is bundled with the project. Launch it within Jupyter by running:

```python
from trend_analysis.gui import launch
launch()
```

The widget reloads the last saved settings from ~/.trend_gui_state.yml and persists new changes after a successful run. Installing the optional ipydatagrid package unlocks spreadsheet-style YAML editing; otherwise the GUI falls back to simpler controls.


## Streamlit App (MVP)

An interactive Streamlit app lets you upload data, configure a simulation, run, and export results.

- Entry point: `streamlit_app/app.py`
- Pages: Upload → Configure → Run → Results → Export

Run it locally:

```bash
./scripts/setup_env.sh
./scripts/run_streamlit.sh
```

Tips
- You can load the included demo dataset from the Upload page.
- The Upload page accepts CSV and Excel files; it validates a Date column and monthly frequency.
- The Results page shows equity and drawdown charts and lets you export a bundle (ZIP) of key outputs.


## Releases

This project uses automated releases to PyPI with semantic versioning.

### For Users
- Install latest stable version: `pip install trend-analysis`
- View releases: [GitHub Releases](https://github.com/stranske/Trend_Model_Project/releases)
- Package on PyPI: [trend-analysis](https://pypi.org/project/trend-analysis/)

### For Maintainers
Create releases by pushing version tags:
```bash
git tag v0.1.0
git push origin v0.1.0
```

The automated workflow will:
- Build distribution packages (wheel + source)
- Test installation 
- Generate changelog from conventional commits
- Publish to PyPI
- Create GitHub release with changelog

For detailed release process documentation, see [docs/release-process.md](docs/release-process.md).

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
