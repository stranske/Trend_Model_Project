# Examples

These examples demonstrate how to use the public `trend_analysis` APIs after
installing the project as a package. Each script lives outside the source tree
so you can explore the behaviors without polluting the root directory.

## Prerequisites

```bash
python -m pip install -e .[app]
```

The editable install exposes the `trend_analysis` package and `trend` CLI used
by these examples. The supported interactive surface is `streamlit_app`.

## Available scripts

| Script | What it shows | How to run |
| --- | --- | --- |
| `examples/demo_robust_weighting.py` | Exercises the robust weighting engines (Ledoit-Wolf/OAS shrinkage, safe-mode fallbacks, logging). | `python examples/demo_robust_weighting.py` |
| `examples/debug_fund_selection.py` | Replays the fund-selection pipeline from `config/portfolio_test.yml`, highlighting missing data, risk-free detection, and final ranked picks. | `python examples/debug_fund_selection.py` |
| `examples/integration_example.py` | Launches the Streamlit app plus the FastAPI/WebSocket proxy, or prints a dry-run summary with `--demo-only`. Requires optional `app` extras. | `python examples/integration_example.py --demo-only` |

> **Tip:** The integration example spawns subprocesses; use the `--demo-only`
> flag first to confirm dependencies, then run the full script when
> `streamlit`, `fastapi`, `uvicorn`, `httpx`, and `websockets` are installed.

Each example relies on the documented CLI and plugin APIs—no manual `sys.path`
manipulation is needed.

## Canonical turnover and report workflows

Turnover constraints are configuration, not a separate executable surface.
`config/long_backtest.yml` sets `portfolio.max_turnover: 0.50` and uses the
checked-in long-backtest dataset:

```bash
trend run -c config/long_backtest.yml \
  --returns data/hedge_fund_returns_with_indexes.csv
```

Generate the supported unified report and summary artifacts through the same
configuration:

```bash
trend report -c config/long_backtest.yml \
  --returns data/hedge_fund_returns_with_indexes.csv \
  --out outputs/long_backtest-report \
  --output outputs/long_backtest-report/report.html
```
