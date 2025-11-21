<!-- bootstrap for codex on issue #3684 -->

## Scope
- [x] Parameters: `per_trade_bps`, `half_spread_bps`, optional `turnover_cap`.
- [x] Compute turnover each rebalance and apply cost drag.
- [x] Persist `turnover` and `cost_drag` series in results.
- [x] Golden test dataset with known weights and expected turnover/cost.

## Tasks
- [x] Add cost/slippage hooks to the backtest loop.
- [x] Record turnover and cost series in results.
- [x] Golden test asserting exact turnover and cost on a tiny toy path.

## Acceptance criteria
- [x] Backtest outputs include turnover and costs.
- [x] Golden test passes with exact expected numbers.
