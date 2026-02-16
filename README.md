# Trend Model Project

A Python-based financial analysis application for volatility-adjusted trend portfolio construction and backtesting. The project provides a command-line interface, interactive Streamlit web application, and Jupyter notebook support for analyzing fund manager performance and constructing optimized portfolios.

## What This Project Does

The Trend Model Project helps you:

- **Analyze fund returns** – Load CSV/Excel data and compute risk-adjusted metrics (CAGR, Sharpe, Sortino, max drawdown, information ratio)
- **Select top performers** – Rank funds by configurable scoring criteria and apply filters
- **Construct portfolios** – Weight selected funds using equal-weight, score-proportional, risk parity, or Bayesian methods
- **Backtest strategies** – Run single-period or multi-period analyses with walk-forward validation
- **Generate reports** – Export results to Excel, CSV, JSON, HTML, and PDF formats

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/stranske/Trend_Model_Project.git
cd Trend_Model_Project

# Set up virtual environment and install dependencies
./scripts/setup_env.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[app]
```

#### LLM/Natural Language Features

LLM extras require Python 3.10+ and Pydantic v2. This project targets Python
3.11+, so the LLM extras are aligned with the base runtime requirement.

```bash
pip install -e ".[llm]"
```

ConfigPatchChain settings can be customized via environment variables:

```bash
export TREND_LLM_MODEL="gpt-4o-mini"
export TREND_LLM_TEMPERATURE="0.2"
```

These values are consumed by `ConfigPatchChain.from_env(...)` to set the model
and temperature used for structured config patch generation.

LangSmith tracing is optional. To enable it, set `LANGSMITH_API_KEY` and
optionally `LANGCHAIN_PROJECT` or `LANGSMITH_PROJECT` to label traces:

```bash
export LANGSMITH_API_KEY="..."
export LANGCHAIN_PROJECT="trend"
```

When the key is present, NL operations send traces to LangSmith automatically.

#### Natural Language Configuration

Use `trend nl` to generate structured config patches from plain-language
instructions. Start with a diff preview and review the full docs before
applying changes:

```bash
trend nl "use risk parity weighting" --in config/demo.yml --diff
```

See `docs/natural-language-config.md` for setup, `docs/nl-cli-reference.md` for
flags, and `docs/nl-troubleshooting.md` for common fixes.

### 2. Verify Installation

```bash
trend --help
```

### 3. Run the Demo

```bash
# Generate synthetic demo data
python scripts/generate_demo.py

# Run analysis with demo configuration
trend run -c config/demo.yml --returns demo/demo_returns.csv
```

### 4. Launch the Web App

```bash
trend app
```

Then open http://localhost:8501 in your browser.

## Usage Options

| Interface | Command | Best For |
|-----------|---------|----------|
| **CLI** | `trend run -c config.yml` | Scripted/automated analysis |
| **Streamlit App** | `trend app` | Interactive exploration |
| **Jupyter GUI** | `from trend_analysis.gui import launch; launch()` | Notebook workflows |

## CLI Migration (Legacy -> trend)

Legacy console scripts still work but forward to `trend` with a deprecation warning.

| Legacy command | Trend equivalent |
|---------------|------------------|
| `trend-analysis` | `trend run` |
| `trend-multi-analysis` | `trend run` (multi-period via config) |
| `trend-model run` | `trend run` (use `--returns` instead of `-i/--input`) |
| `trend-model gui` | `trend app` |
| `trend-model --check` | `trend check` |
| `trend-app` | `trend app` |
| `trend-run` | `trend report` |
| `trend-quick-report` | `trend quick-report` |

### Command-Line Examples

```bash
# Basic analysis with config file
trend run -c config/demo.yml --returns data/returns.csv

# Use a preset strategy
trend run -c config/demo.yml --returns data/returns.csv --preset conservative

# Generate a report from previous results
trend report --last-run demo/portfolio_test_results/last_run_results.json

# Run stress test scenarios
trend stress -c config/demo.yml

# Walk-forward analysis
python scripts/walk_forward.py --config config/walk_forward.yml
```

### Monte Carlo Visualization (`trend mc viz`)

```bash
trend mc viz --bundle <path> --out <dir> --charts fan,path_dist,risk_return --html --json --png
```

- `--bundle`: Path to the Monte Carlo bundle directory used as input.
- `--out`: Output directory where `plots/` is created (for example `<out>/plots/`).
- `--charts`: Comma-separated chart names to render, for example `fan,path_dist,risk_return`.
- `--html`: Write HTML chart files to `<out>/plots/*.html`.
- `--json`: Write JSON chart files to `<out>/plots/*.json`.
- `--png`: Write PNG chart files to `<out>/plots/*.png`.

NAV-path selection options:

- `--fold <id>`: Use fold-exported `nav_paths_fold_<id>.parquet` when the bundle comes from a fold run.
- `--nav-paths <file>`: Use an explicit `*.parquet` NAV-path file instead of `<bundle>/nav_paths.parquet`.

### NAV-Path Utilities and `mc viz` Contracts

Use the NAV-path helpers from `trend.mc.io` instead of re-implementing loading/validation logic:

```python
from trend.mc.io import (
    MISSING_NAV_PATHS_RAISE,
    load_nav_paths,
    validate_nav_paths_df,
)

nav_paths = load_nav_paths(
    "tests/fixtures/mc_bundle",
    missing_parquet=MISSING_NAV_PATHS_RAISE,
)
validated = validate_nav_paths_df(nav_paths)

# Canonical `nav_paths` shape:
# - Wide DataFrame
# - Datetime-like index
# - One column per simulated path (optionally MultiIndex columns with levels like (path, asset="NAV"))
# - Values normalized so each path starts at 1.0

# Optional: convert wide NAV paths to long form for downstream charts/diagnostics.
from trend_analysis.viz.adapters import make_paths

paths_long = make_paths(validated)
```

Missing `nav_paths.parquet` contract:

- `load_nav_paths(..., missing_parquet="raise")` raises `MCNavPathsIOError` with message prefix `Missing required nav_paths.parquet file in MC bundle:`.
- `load_nav_paths(..., missing_parquet="return-none")` emits `MCNavPathsWarning` with message containing `Missing optional nav_paths.parquet file in MC bundle:` and `Continuing without NAV-path data.` and returns `None`.
- `trend mc viz` raises `MCNavPathsIOError` when required charts are selected without NAV paths, with message containing `require nav_paths.parquet in the MC bundle`.

HTML/PNG selection and markers:

- `trend mc viz` uses one selected chart set for both HTML and PNG output modes.
- Each generated chart HTML includes an exact marker comment in this format: `<!-- mc-viz-chart:<chart_id> -->`.
- For the default charts, marker strings are:
  - `<!-- mc-viz-chart:fan -->`
  - `<!-- mc-viz-chart:path_dist -->`
  - `<!-- mc-viz-chart:risk_return -->`

Artifact extraction behavior:

- Use the single public extractor `trend_analysis.viz.artifacts.extract_bundle_zip(...)`.
- Extraction is deterministic and non-destructive; filename collisions are disambiguated with stable numeric suffixes (`-1`, `-2`, ...), for example `risk_return.json` and `risk_return-1.json`.

Output directory creation semantics:

- `trend mc viz` creates parent output directories automatically (`Path(...).mkdir(parents=True, exist_ok=True)`) before writing chart artifacts.
- Artifact extraction also creates required parent directories automatically before writing extracted files.
- Example nested output path that works without pre-creating directories:

```bash
trend mc viz --bundle tests/fixtures/mc_bundle --out tmp/reports/2026/02/12/mc --charts fan --html --json
```

## Configuration

Analysis parameters are controlled via YAML configuration files. The key sections are:

```yaml
data:
  returns_file: "data/returns.csv"
  risk_free_column: "T-Bill"          # Optional cash proxy
  missing_policy: "ffill"             # Handle gaps: drop, ffill, or zero

portfolio:
  selection_mode: "rank"              # all, random, manual, or rank
  top_n: 10                           # Number of funds to select
  weighting:
    method: "equal"                   # equal, score_prop, risk_parity, hrp, etc.

vol_adjust:
  target_vol: 0.10                    # 10% annualized volatility target

output:
  format: "excel"                     # excel, csv, or json
  path: "outputs/results"
```

See `config/defaults.yml` for the complete schema and `config/presets/` for ready-made strategies.

## Environment Variables

Anthropic keys resolve in order, with `CLAUDE_API_STRANSKE` as the canonical primary and
`ANTHROPIC_API_KEY` used only as a fallback when the primary is unset.

- `CLAUDE_API_STRANSKE`: Canonical Anthropic API key (primary).
- `ANTHROPIC_API_KEY`: Anthropic API key (fallback, used only if CLAUDE_API_STRANSKE is unset).

## Project Structure

```
Trend_Model_Project/
├── src/trend_analysis/     # Core analysis package
│   ├── pipeline.py         # Main orchestration
│   ├── metrics.py          # Financial metrics
│   ├── export/             # Output formatters
│   └── config/             # Configuration models
├── src/trend_portfolio_app/ # Streamlit application
├── streamlit_app/          # Streamlit pages
├── config/                 # YAML configuration files
│   ├── defaults.yml
│   └── presets/            # Conservative, balanced, aggressive
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── demo/                   # Generated demo datasets
```

## Documentation

| Document | Purpose |
|----------|---------|
| **[User Guide](docs/UserGuide.md)** | Complete feature walkthrough with examples |
| **[README_APP.md](README_APP.md)** | Streamlit app layout and features |
| **[README_DATA.md](README_DATA.md)** | Demo data provenance and limitations |
| **[docs/INDEX.md](docs/INDEX.md)** | Full documentation index |
| **[docs/CLI.md](docs/CLI.md)** | Command-line interface reference |
| **[docs/ConfigMap.md](docs/ConfigMap.md)** | Configuration parameter reference |
| **[docs/PresetStrategies.md](docs/PresetStrategies.md)** | Strategy preset descriptions |
| **[docs/natural-language-config.md](docs/natural-language-config.md)** | Natural language config guide |

## Key Features

### Selection Modes

- **all** – Include every fund in the portfolio
- **rank** – Select top N funds by score
- **random** – Randomly sample funds (for Monte Carlo analysis)
- **manual** – Hand-pick funds via GUI

### Weighting Methods

| Method | Description |
|--------|-------------|
| `equal` | Simple 1/N allocation |
| `score_prop` | Weights proportional to scores |
| `score_prop_bayes` | Bayesian shrinkage of scores |
| `adaptive_bayes` | Cross-period learning |
| `risk_parity` | Equal risk contribution |
| `hrp` | Hierarchical risk parity |

### Risk Controls

- Volatility targeting with configurable lookback windows
- Maximum weight constraints per asset
- Group-level allocation caps
- Turnover limits and transaction cost modeling

### Output Formats

- **Excel** – Formatted workbook with summary sheet
- **CSV** – Machine-readable metrics
- **JSON** – Structured data for programmatic consumption
- **HTML/PDF** – Tear sheets and reports (via `trend report`)

## Development

### Run Tests

```bash
./scripts/run_tests.sh
```

### Validation

```bash
# Quick check during development
./scripts/dev_check.sh --fix

# Comprehensive pre-commit validation
./scripts/validate_fast.sh --fix

# Full CI-equivalent check
./scripts/check_branch.sh --fast --fix
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## File Inventory

| File | Purpose |
|------|---------|
| `Agents.md` | Guard-rails and workflow guidance for contributors |
| `CHANGELOG.md` | Release notes |
| `CONTRIBUTING.md` | Contribution guidelines |
| `DEPENDENCY_QUICKSTART.md` | Dependency setup cheat sheet |
| `DOCKER_QUICKSTART.md` | Docker usage guide |
| `SECURITY.md` | Security policy |

## License

[MIT License](LICENSE)
