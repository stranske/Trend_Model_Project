# Non-Bayesian Rebalancing Strategies

This document explains the non-Bayesian rebalancing strategies available in the Trend Analysis Project (Phase 1 implementation).

## Overview

The rebalancing module provides strategies for determining when and how to adjust portfolio weights based on drift, time intervals, and other criteria. These strategies can be composed with Bayesian weighting methods when `bayesian_only=false`.

## Configuration

Add rebalancing configuration to your YAML config file under `portfolio.rebalance`:

```yaml
portfolio:
  rebalance:
    bayesian_only: false              # Set to false to enable non-Bayesian strategies
    strategies:                       # Applied in sequence
      - drift_band                    
      - periodic_rebalance           
    params:
      drift_band:
        band_pct: 0.03                # Max drift as absolute percentage (3%)
        min_trade: 0.005              # Min trade size as absolute weight (0.5%)  
        mode: partial                 # "partial" or "full" rebalancing
      periodic_rebalance:
        interval: 1                   # Rebalance every N periods
```

## Strategies

### 1. Drift Band Strategy (`drift_band`)

Rebalances when portfolio weights drift outside specified bands around target weights.

**Parameters:**
- `band_pct` (float, default 0.03): Maximum allowed drift as absolute percentage. For example, 0.03 means 3% absolute drift from target weight.
- `min_trade` (float, default 0.005): Minimum trade size as absolute weight. Trades smaller than this are ignored.
- `mode` (str, default "partial"): Rebalancing mode
  - `"partial"`: Only trade the excess drift above the band edge
  - `"full"`: Fully rebalance back to exact target weights

**Behavior:**
- **No Trigger**: When all asset weights are within the band and above minimum trade size
- **Partial Mode**: Trades only the excess drift above the band boundary
- **Full Mode**: Rebalances completely back to target weights when any asset exceeds the band

**Example:**
```python
# Target weights: A=50%, B=50%  
# Current weights: A=56%, B=44% (6% drift)
# band_pct=0.03, mode="partial"

# Trigger: Yes (6% > 3% band)
# Result: A=53%, B=47% (trade 3% back to band edge)
```

### 2. Periodic Rebalance Strategy (`periodic_rebalance`)

Rebalances at regular time intervals regardless of current drift.

**Parameters:**
- `interval` (int, default 1): Number of periods between rebalances. Must be >= 1.

**Behavior:**
- **First Period**: Always triggers rebalancing
- **Subsequent Periods**: Triggers every `interval` periods since last rebalance
- **Rebalancing**: Always fully rebalances to target weights

**Example:**
```python
# interval=3
# Period 1: Rebalance (first period)
# Period 2-3: Hold  
# Period 4: Rebalance (1 + 3)
# Period 5-6: Hold
# Period 7: Rebalance (4 + 3)
```

## Engine Behavior

### Bayesian-Only Mode (`bayesian_only: true`)
- Skips all non-Bayesian strategies
- Returns target weights directly (assumed to be from Bayesian weighting)
- No additional trades generated

### Composed Mode (`bayesian_only: false`)
- Applies strategies in the configured sequence
- Each strategy operates on the output of the previous strategy
- Final result can differ significantly from pure Bayesian weights

### Strategy Sequencing

When multiple strategies are configured, they are applied in sequence:

1. Start with current portfolio weights
2. Apply first strategy → intermediate weights + trades
3. Apply second strategy to intermediate weights → final weights + more trades
4. Return combined trades and final weights

**Example with `drift_band` + `periodic_rebalance`:**
```python
# Period 1, interval=1 for periodic
current = {"A": 0.55, "B": 0.45}  # 5% drift from equal
target = {"A": 0.50, "B": 0.50}

# Step 1: drift_band (partial, band=0.03)
# Result: {"A": 0.52, "B": 0.48}  # Trade 2% back to band edge

# Step 2: periodic_rebalance (interval=1, always triggers)  
# Result: {"A": 0.50, "B": 0.50}  # Full rebalance to target
```

## Integration Points

### With Bayesian Weighting
- Bayesian weighting methods compute "target" weights
- Non-Bayesian strategies determine how to transition from current to target weights
- Set `bayesian_only: false` to enable this composition

### With Multi-Period Engine
- Rebalancing strategies integrate with the existing multi-period framework
- Strategies receive period numbers for time-based triggers
- State is maintained across periods (e.g., last rebalance period)

## Default Configuration

If not specified, the following defaults are used:

```yaml
portfolio:
  rebalance:
    bayesian_only: true               # Disabled by default
    strategies: ["drift_band"]        # Single strategy
    params:
      drift_band:
        band_pct: 0.03
        min_trade: 0.005  
        mode: partial
      periodic_rebalance:
        interval: 1
```

## Usage Examples

### Conservative Rebalancing
Only rebalance when significant drift occurs:
```yaml
portfolio:
  rebalance:
    bayesian_only: false
    strategies: ["drift_band"]
    params:
      drift_band:
        band_pct: 0.05    # Wide band (5%)
        min_trade: 0.01   # Larger minimum trade (1%)
        mode: partial
```

### Regular Rebalancing
Rebalance quarterly regardless of drift:
```yaml
portfolio:
  rebalance:
    bayesian_only: false  
    strategies: ["periodic_rebalance"]
    params:
      periodic_rebalance:
        interval: 3       # Every 3 periods (quarters if monthly data)
```

### Combined Approach
Drift bands with periodic full rebalancing:
```yaml
portfolio:
  rebalance:
    bayesian_only: false
    strategies: 
      - drift_band
      - periodic_rebalance  
    params:
      drift_band:
        band_pct: 0.02    # Tight drift control
        mode: partial
      periodic_rebalance:
        interval: 12      # Annual full rebalancing
```

## Output Structure

The rebalancing engine returns a `RebalanceResult` containing:

- `realized_weights`: Final portfolio weights after rebalancing
- `trades`: List of `RebalanceEvent` objects with trade details
- `should_rebalance`: Boolean indicating if any rebalancing occurred

Each `RebalanceEvent` contains:
- `symbol`: Asset identifier
- `current_weight`: Weight before trade
- `target_weight`: Desired target weight
- `trade_amount`: Actual trade amount (positive = buy, negative = sell)
- `reason`: String describing why the trade was made

## Performance Considerations

- **Drift Band**: Computational cost scales with number of assets
- **Periodic**: Minimal computational overhead
- **State Management**: Strategies maintain minimal state across periods
- **Memory**: Trade history is not persisted across periods

## Future Extensions

The modular design allows for easy addition of new strategies:

- `turnover_cap`: Limit total portfolio turnover
- `vol_target_rebalance`: Rebalance based on volatility targets  
- `drawdown_guard`: Defensive rebalancing during drawdowns

These strategies are already referenced in the Streamlit UI but not yet implemented.