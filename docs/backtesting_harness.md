# Honest Backtesting Harness

The honest backtesting harness introduced for Issue #1681 adds a reusable
`run_backtest` function under `trend_analysis.backtesting`.  The helper couples
walk-forward portfolio evaluation with:

* **Fixed rebalance calendars** – pass any pandas frequency string (e.g. `"M"`
  or `"W"`) and the harness will only adjust weights on the corresponding
  calendar boundaries.
* **Rolling *and* expanding windows** – switch between the two behaviours with
  the `window_mode` flag without changing your strategy code.
* **Basis-point transaction costs** – supply `transaction_cost_bps` to apply
  fees on the absolute change in target weights at each rebalance.
* **Rich performance analytics** – the returned `BacktestResult` exposes the
  equity curve, turnover, transaction-cost ledger, rolling Sharpe ratios, and
  drawdown time series alongside summary statistics (CAGR, volatility, Sortino,
  Calmar, max drawdown, Sharpe).

Downstream integrations can serialise the result via
`BacktestResult.to_json()`, which emits a JSON payload containing the summary
metrics, rolling Sharpe series, drawdown path, rebalance calendar, turnover,
transaction-cost ledger, sparse weight history, and the full set of training
window boundaries so that UI layers can present the walk-forward evaluation
without bespoke post-processing or duplicated window bookkeeping.

See `tests/backtesting/test_harness.py` for end-to-end usage examples covering
window switching and transaction-cost verification.
