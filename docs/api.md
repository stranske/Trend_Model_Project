# API Reference

This document provides an overview of the Trend Model Project's programmatic interfaces.

## Python API

### Core Pipeline

```python
from trend_analysis.pipeline import run

# Run analysis with config file
results = run(config_path="config/demo.yml")

# Run with config object
from trend_analysis.config import load
config = load("config/demo.yml")
results = run(config=config)
```

### Configuration

```python
from trend_analysis.config import load, Config

# Load from file
config = load("config/defaults.yml")

# Access configuration values
print(config.data.csv_path)
print(config.portfolio.top_n)
```

### Metrics

```python
from trend_analysis.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_cagr,
)

# Calculate metrics on return series
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
max_dd = calculate_max_drawdown(returns)
cagr = calculate_cagr(returns, periods_per_year=12)
```

### Export

```python
from trend_analysis.export import export_to_excel, export_to_csv, export_to_json

# Export results to various formats
export_to_excel(results, "output.xlsx")
export_to_csv(results, "output")
export_to_json(results, "output.json")
```

## REST API

The project includes a FastAPI server for programmatic access.

### Starting the Server

```bash
uvicorn trend_analysis.api_server:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Run analysis with config |
| `/docs` | GET | OpenAPI documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/demo.yml"}'
```

## CLI Interface

See [CLI.md](CLI.md) for command-line interface documentation.

## Module Structure

```
src/trend_analysis/
├── __init__.py          # Package exports
├── config/              # Configuration loading
├── core/                # Core algorithms
├── engine/              # Analysis engine
├── export/              # Export functionality
├── metrics.py           # Metric calculations
├── pipeline.py          # Main pipeline
├── data.py              # Data loading/processing
└── api_server/          # REST API
```

## See Also

- [UserGuide.md](UserGuide.md) - User documentation
- [CLI.md](CLI.md) - Command-line interface
- [config.md](config.md) - Configuration reference
