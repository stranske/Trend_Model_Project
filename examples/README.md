# Examples

These examples demonstrate how to use the public `trend_analysis` APIs after
installing the project as a package. Each script lives outside the source tree
so you can explore the behaviors without polluting the root directory.

## Prerequisites

```bash
python -m pip install -e .[app]
```

The editable install exposes the `trend_analysis`, `trend_portfolio_app` and
`trend` CLI entry points that every example depends on.

## Available scripts

| Script | What it shows | How to run |
| --- | --- | --- |
| `examples/demo_robust_weighting.py` | Exercises the robust weighting engines (Ledoit-Wolf/OAS shrinkage, safe-mode fallbacks, logging). | `python examples/demo_robust_weighting.py` |
| `examples/demo_turnover_cap.py` | Backwards-compatible wrapper that delegates to the `trend` CLI with the demo config, keeping the historical turnover-cap shortcut alive. | `python examples/demo_turnover_cap.py` |
| `examples/debug_fund_selection.py` | Replays the fund-selection pipeline from `config/portfolio_test.yml`, highlighting missing data, risk-free detection, and final ranked picks. | `python examples/debug_fund_selection.py` |
| `examples/portfolio_analysis_report.py` | Deprecated script that forwards directly to `trend.cli:main`, helping teams migrate to the CLI while retaining the old entry point. | `python examples/portfolio_analysis_report.py --help` |
| `examples/integration_example.py` | Launches the Streamlit app plus the FastAPI/WebSocket proxy, or prints a dry-run summary with `--demo-only`. Requires optional `app` extras. | `python examples/integration_example.py --demo-only` |

> **Tip:** The integration example spawns subprocesses; use the `--demo-only`
> flag first to confirm dependencies, then run the full script when
> `streamlit`, `fastapi`, `uvicorn`, `httpx`, and `websockets` are installed.

Each example relies on the documented CLI and plugin APIs—no manual `sys.path`
manipulation is needed.
