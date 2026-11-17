# PR #6231 verification log

## Regime label → cost regime mapping

Implemented `_resolve_regime_labels` mapping when a `CostProcess` is active:

- `RegimeSettings.risk_on_label` → `calm` (if `calm` is a configured cost regime)
- `RegimeSettings.risk_off_label` → `stress` (if `stress` is configured)
- Labels that already match cost regime keys are left unchanged.

## Tests

```text
python3 -m pytest tests/monte_carlo/test_costs.py::test_real_runner_maps_risk_off_to_stress_costs -q -m "not slow"  # PASS
python3 -m pytest tests/monte_carlo/test_costs.py -q -m "not slow"  # 30 passed
```

## Deliberate break → revert

Removed Risk-On→calm / Risk-Off→stress mapping in `_resolve_regime_labels` locally:

- `test_real_runner_maps_risk_off_to_stress_costs` **FAILED** (`costs.regimes` stayed `Risk-On` / `Risk-Off`; stress 100bps not applied)

Restored mapping:

- `test_real_runner_maps_risk_off_to_stress_costs` **PASSED**