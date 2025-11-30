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
