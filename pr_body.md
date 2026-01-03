<!-- pr-preamble:start -->
> **Source:** Issue #4142

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Rebalancing strategies can produce weights that don't sum to 1.0, leaving "missing mass" as implicit cash without explicit modeling. This silently makes the assumption that cash earns 0% return and has 0 financing cost—a modeling choice that can significantly bias results in any period where the risk-free rate isn't near zero.

The issue spans multiple strategy implementations:
- **TurnoverCapStrategy**: After partial trade execution, clips negatives but doesn't normalize or add explicit cash
- **DriftBandStrategy** (partial mode): Overwrites some names with target weights, changing total weight
- **VolTargetRebalanceStrategy**: Scales weights by leverage without financing/carry modeling
- **DrawdownGuardStrategy**: Scales weights down leaving implicit cash without defining what cash earns

#### Tasks
- [ ] Define a `CashPolicy` dataclass with fields: `explicit_cash: bool`, `cash_return_source: str` (config key or literal), `normalize_weights: bool`
- [ ] Add optional `cash_policy` parameter to `Rebalancer.apply()` signature with backward-compatible default
- [ ] Update `TurnoverCapStrategy._apply_turnover_cap()` to optionally add explicit CASH line for unexecuted weight mass
- [ ] Update `DriftBandStrategy.apply()` to track weight sum delta and either normalize or add CASH
- [ ] Update `VolTargetRebalanceStrategy.apply()` to return `(weights * lev, cost)` with optional financing spread
- [ ] Update `DrawdownGuardStrategy.apply()` to add CASH line when scaling down instead of leaving mass implicit
- [ ] Add integration test verifying all strategies return weights summing to 1.0 (or 1.0 + CASH)
- [ ] Document cash modeling behavior in strategy docstrings

#### Acceptance criteria
- [ ] All rebalancing strategies return weights summing to exactly 1.0 when `normalize_weights=True`
- [ ] When `explicit_cash=True`, unallocated mass appears as a CASH line in returned weights
- [ ] Existing tests pass without modification (backward compatibility maintained)
- [ ] New unit tests cover all cash policy combinations for each strategy
- [ ] Strategy docstrings document cash behavior and assumptions

<!-- auto-status-summary:end -->
