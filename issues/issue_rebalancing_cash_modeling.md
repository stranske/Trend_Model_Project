# Make cash a first-class asset in rebalancing strategies

## Why

Rebalancing strategies can produce weights that don't sum to 1.0, leaving "missing mass" as implicit cash without explicit modeling. This silently makes the assumption that cash earns 0% return and has 0 financing cost—a modeling choice that can significantly bias results in any period where the risk-free rate isn't near zero.

The issue spans multiple strategy implementations:
- **TurnoverCapStrategy**: After partial trade execution, clips negatives but doesn't normalize or add explicit cash
- **DriftBandStrategy** (partial mode): Overwrites some names with target weights, changing total weight
- **VolTargetRebalanceStrategy**: Scales weights by leverage without financing/carry modeling
- **DrawdownGuardStrategy**: Scales weights down leaving implicit cash without defining what cash earns

## Scope

Unify cash handling across all rebalancing strategies to make implicit cash explicit and model cash returns consistently.

## Non-Goals

- Implementing full portfolio financing models (margin accounts, repo, etc.)
- Changing the existing weight policy architecture
- Backward-incompatible changes to public APIs

## Tasks

- [ ] Define a `CashPolicy` dataclass with fields: `explicit_cash: bool`, `cash_return_source: str` (config key or literal), `normalize_weights: bool`
- [ ] Add optional `cash_policy` parameter to `Rebalancer.apply()` signature with backward-compatible default
- [ ] Update `TurnoverCapStrategy._apply_turnover_cap()` to optionally add explicit CASH line for unexecuted weight mass
- [ ] Update `DriftBandStrategy.apply()` to track weight sum delta and either normalize or add CASH
- [ ] Update `VolTargetRebalanceStrategy.apply()` to return `(weights * lev, cost)` with optional financing spread
- [ ] Update `DrawdownGuardStrategy.apply()` to add CASH line when scaling down instead of leaving mass implicit
- [ ] Add integration test verifying all strategies return weights summing to 1.0 (or 1.0 + CASH)
- [ ] Document cash modeling behavior in strategy docstrings

## Acceptance Criteria

- [ ] All rebalancing strategies return weights summing to exactly 1.0 when `normalize_weights=True`
- [ ] When `explicit_cash=True`, unallocated mass appears as a CASH line in returned weights
- [ ] Existing tests pass without modification (backward compatibility maintained)
- [ ] New unit tests cover all cash policy combinations for each strategy
- [ ] Strategy docstrings document cash behavior and assumptions

## Implementation Notes

Files to modify:
- `src/trend_analysis/rebalancing/strategies.py` - All strategy classes
- `src/trend_analysis/rebalancing/__init__.py` - Export `CashPolicy`
- `tests/test_rebalancing_strategies.py` - Add cash modeling tests

The `apply_weight_policy()` in `src/trend_analysis/portfolio/weight_policy.py` already supports a `"cash"` mode that leaves weights unnormalized with residual as cash buffer. Rebalancing should follow the same philosophy.

Reference: Lines 127-137 in `strategies.py` show `TurnoverCapStrategy._apply_turnover_cap()` doing `new_weights.clip(lower=0.0)` without normalization.
