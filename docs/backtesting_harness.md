# Honest Backtesting Harness

The honest backtesting harness introduced for Issue #1681 adds a reusable
`run_backtest` function under `trend_analysis.backtesting`.  The helper couples
walk-forward portfolio evaluation with:

* **Fixed rebalance calendars** – pass any pandas frequency string (e.g. `"M"`
  or `"W"`) and the harness will only adjust weights on the corresponding
  calendar boundaries.
* **Rolling *and* expanding windows** – switch between the two behaviours with
  the `window_mode` flag without changing your strategy code.
* **Basis-point transaction & slippage costs** – supply `transaction_cost_bps`
  together with optional `slippage_bps`, or pass a
  `trend_analysis.costs.CostModel` instance for per-turnover deductions on
  every rebalance.
* **Minimum-trade bands** – use `min_trade` to skip rebalance instructions
  whose total absolute weight change falls below the specified threshold,
  preventing micro-churn from leaking into the turnover ledger.
* **Rich performance analytics** – the returned `BacktestResult` exposes the
  equity curve, turnover, per-period turnover series, transaction-cost ledger,
  rolling Sharpe ratios, and
  drawdown time series alongside summary statistics (CAGR, volatility, Sortino,
  Calmar, max drawdown, Sharpe).

Downstream integrations can serialise the result via
`BacktestResult.to_json()`, which emits a JSON payload containing the summary
metrics, realised return path, rolling Sharpe series, drawdown profile,
rebalance calendar, turnover, transaction-cost ledger, sparse weight history,
and the full set of training window boundaries so that UI layers can present
the walk-forward evaluation without bespoke post-processing or duplicated
window bookkeeping.

See `tests/backtesting/test_harness.py` for end-to-end usage examples covering
window switching and deterministic cost-model verification.

## Bootstrap uncertainty bands

Call `trend_analysis.backtesting.bootstrap_equity(result, n=500, block=20)` to
estimate a median equity path together with the 5th and 95th percentile bands
for a realised backtest.  The helper consumes a `BacktestResult`, applies a
circular block bootstrap to the `returns` series, and returns a DataFrame
indexed like `result.equity_curve` with `p05`, `median`, and `p95` columns.

Key parameters:

* `n`: number of sampled paths (at least 1).
* `block`: bootstrap block length in periods (must be positive).
* `random_state`: optional seed/generator so tests and exports can reproduce
  deterministic bands.

Any leading `NaN` values in the original return series are reinserted so charts
and exports align with the point where the strategy becomes live.  The helper
also honours the realised equity scale (e.g. starting capital of 100) so the
resulting quantiles overlay correctly on absolute-dollar curves.  See
`tests/backtesting/test_bootstrap.py` for worked examples and edge-case
coverage.
