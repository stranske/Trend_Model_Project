# Temporary Inventory: `costs:` YAML Blocks in `MonteCarlo.md`

Purpose: capture current `costs:` example structures before canonical-schema cleanup.

## Block 1 (`docs/phase-3/MonteCarlo.md:376`)

Context: canonical example.

```yaml
costs:
  kind: regime_stochastic
  default_regime: calm
  calm:
    trade_cost_bps:
      dist: lognormal
      mean: 6
      sigma: 0.25
  stress:
    trade_cost_bps:
      dist: lognormal
      mean: 18
      sigma: 0.35
    slippage_multiplier: 1.5
```

Top-level keys under `costs`: `kind`, `default_regime`, `calm`, `stress`.

## Block 2 (`docs/phase-3/MonteCarlo.md:403`)

Context: legacy `regimes` mapping format.

```yaml
costs:
  default_regime: calm
  regimes:
    calm:
      distribution:
        kind: fixed
        value: 6
    stress:
      distribution:
        kind: fixed
        value: 18
      slippage_multiplier: 1.5
```

Top-level keys under `costs`: `default_regime`, `regimes`.
Nested legacy keys: `distribution.kind`, `distribution.value`.

## Block 3 (`docs/phase-3/MonteCarlo.md:419`)

Context: alias example.

```yaml
costs:
  regimes:
    calm:
      trade_cost_bps:
        dist: normal
        mean: 6
        std: 1.5
```

Top-level keys under `costs`: `regimes`.
Nested keys include alias/variant fields: `dist`, `std`.

## Block 4 (`docs/phase-3/MonteCarlo.md:430`)

Context: numeric shorthand.

```yaml
costs:
  regimes:
    calm: 6
    stress: 18
```

Top-level keys under `costs`: `regimes`.
Nested regime values are numeric scalars.

## Block 5 (`docs/phase-3/MonteCarlo.md:580`)

Context: complete scenario schema example.

```yaml
costs:
  kind: regime_stochastic
  default_regime: calm
  calm:
    trade_cost_bps:
      dist: lognormal
      mean: 6
      sigma: 0.25
  stress:
    trade_cost_bps:
      dist: lognormal
      mean: 18
      sigma: 0.35
    slippage_multiplier: 1.5
```

Top-level keys under `costs`: `kind`, `default_regime`, `calm`, `stress`.
