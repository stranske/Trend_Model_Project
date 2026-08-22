# Usage Guide

Quick reference for common Trend Model Project operations.

## Analysis

### Command Line

```bash
# Basic analysis with config file
PYTHONPATH="./src" python -m trend.cli run -c config/demo.yml

# Using environment variable
TREND_CFG=config/demo.yml PYTHONPATH="./src" python -m trend.cli run

# With default config
PYTHONPATH="./src" python -m trend.cli run
```

### Streamlit App

```bash
# Launch the web interface
./scripts/run_streamlit.sh

# Or directly
streamlit run streamlit_app/app.py
```

Access at http://localhost:8501

### CLI Tools

```bash
# Launch Streamlit
trend app

# Run analysis from TOML config
trend run -c config/trend.toml
```

## Data

### Generate Demo Data

```bash
python scripts/generate_demo.py
```

Creates `demo/demo_returns.csv` and `demo/demo_returns.xlsx`.

### Input Format

CSV files must include:
- `Date` column (YYYY-MM-DD format)
- One column per asset with return values

## Configuration

### Config Files

| File | Purpose |
|------|---------|
| `config/defaults.yml` | Default configuration |
| `config/demo.yml` | Demo scenario |
| `config/presets/*.yml` | Risk profile presets |

### Key Settings

```yaml
data:
  csv_path: demo/demo_returns.csv
  date_column: Date

portfolio:
  top_n: 8
  lookback: 12

export:
  directory: outputs
  filename: analysis
  formats: [xlsx, csv, json]
```

## Testing

```bash
# Full test suite
./scripts/run_tests.sh

# Quick validation
./scripts/dev_check.sh --fix

# Specific tests
pytest tests/test_pipeline.py -v
```

## Docker

```bash
# Run container
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest

# Build locally
docker build -t trend-model .
docker run -p 8501:8501 trend-model
```

## See Also

- [quickstart.md](quickstart.md) - Getting started tutorial
- [UserGuide.md](UserGuide.md) - Comprehensive guide
- [CLI.md](CLI.md) - CLI reference
- [config.md](config.md) - Configuration reference
