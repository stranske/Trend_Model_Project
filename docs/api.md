# API Reference

This document provides an overview of the Trend Model Project's programmatic interfaces.

## Python API

### Core Pipeline

Run these examples from the repository root with the package and export extras
installed. `run(cfg)` returns a pandas DataFrame of out-of-sample metrics;
`run_full(cfg)` is the separate API for detailed diagnostics.

```python
from trend_analysis.config import load
from trend_analysis.pipeline import run

config = load("config/demo.yml")
results = run(config)
```

### Configuration

Configuration sections expose dictionary values. The demo configuration resolves
its input CSV relative to the configuration file.

```python
print(config.data["csv_path"])
print(config.portfolio["selection_mode"])
```

### Metrics

The demo file contains periodic returns, so no price-to-return conversion is
needed. The risk-free argument below is a per-period return, not an annual rate.

```python
import pandas as pd
from trend_analysis.metrics import annual_return, max_drawdown, sharpe_ratio

returns = pd.read_csv("demo/demo_returns.csv")["Mgr_01"]
sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=12)
max_dd = max_drawdown(returns)
annualized = annual_return(returns, periods_per_year=12)
```

### Export

Exporters accept a mapping of sheet/dataset names to DataFrames. CSV and JSON
exporters append each dataset name to the output prefix: these calls produce
`output.xlsx`, `output_metrics.csv` and `output_metrics.json`.

```python
from trend_analysis.export import export_to_csv, export_to_excel, export_to_json

frames = {"metrics": results}
export_to_excel(frames, "output.xlsx")
export_to_csv(frames, "output")
export_to_json(frames, "output")
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
| `/` | GET | API metadata (version, links) |
| `/config/patch` | POST | Apply a config patch with risk confirmation |
| `/config/patch/preview` | POST | Preview a config patch (returns updated config + diff) |
| `/docs` | GET | OpenAPI documentation |

The REST surface is scoped to config-patch preview and apply. Running an
analysis is not exposed over HTTP — use the [`trend run`](CLI.md) CLI
(`trend run -c <config>`) or the Python API at the top of this document
instead.

### Example Request

The `patch` body must match the
[`ConfigPatch`](../src/trend_analysis/config/patch.py) schema: a list of
`operations` (each with `op`, `path`, and a `value` for `set`/`append`/`merge`)
plus a non-empty `summary`. Extra keys at the patch root are rejected. Patches
that trigger risk flags (constraint removals, leverage increases, validation
removals, broad-scope edits) or `needs_review` must be sent with
`confirm_risky: true`.

```bash
curl -X POST http://localhost:8000/config/patch/preview \
  -H "Content-Type: application/json" \
  -d '{
        "config": {"analysis": {"top_n": 10}},
        "patch": {
          "operations": [
            {"op": "set", "path": "analysis.top_n", "value": 12}
          ],
          "summary": "Update selection count to 12.",
          "needs_review": true
        },
        "confirm_risky": true
      }'
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
