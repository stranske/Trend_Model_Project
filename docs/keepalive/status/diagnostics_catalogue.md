# Diagnostics early-exit catalogue

Initial sweep of silent `None` returns across the diagnostics surface. Each entry pairs the location with a proposed reason code/message so we can replace bare `None` values with diagnostics payloads in follow-up patches.

## Reporting helpers
- `src/trend/reporting/unified.py::_build_backtest` — returns `None` when portfolio time series cannot be coerced or is empty. Proposed reason code: `NO_PORTFOLIO_SERIES`; message: "Backtest results missing portfolio returns for report rendering." 
- `src/trend/reporting/quick_summary.py::_equity_drawdown_chart` — returns `None` when equity curve is empty. Proposed reason code: `NO_RETURNS_SERIES`; message: "No portfolio returns available to draw equity curve." 
- `src/trend/reporting/quick_summary.py::_turnover_chart` — returns `None` when turnover diagnostics are empty. Proposed reason code: `NO_TURNOVER_SERIES`; message: "Turnover diagnostics missing or empty; chart skipped." 
- `src/trend/reporting/quick_summary.py::_maybe_datetime_index` / `_extract_returns` / `_extract_turnover` — implicitly yield empty series when conversions fail. Proposed reason code: `MISSING_TIMESTAMPS`; message: "Input series lacks convertible timestamp index." 

## CLI glue
- `src/trend/cli.py::_maybe_write_turnover_csv` — returns `None` for non-mapping diagnostics, non-numeric turnover, or empty turnover series. Proposed reason code: `NO_TURNOVER_EXPORT`; message: "Turnover diagnostics absent or non-numeric; skipping CSV export." 
- `src/trend/cli.py::_persist_turnover_ledger` — returns `None` when no turnover payload is present. Proposed reason code: `NO_TURNOVER_LEDGER`; message: "No turnover diagnostics captured for ledger persistence." 
- `src/trend/cli.py::_init_perf_logger` — returns `None` when perf logging is disabled or fails to initialise. Proposed reason code: `PERF_LOG_DISABLED`; message: "Performance logging disabled or could not be initialised." 

## Next steps
- Thread the reason codes above into structured diagnostic objects instead of bare `None` returns.
- Update CLI/Reporting entry points to propagate and surface these diagnostics.
- Backfill targeted tests that assert the diagnostics objects are emitted for each early-exit case.
