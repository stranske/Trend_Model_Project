# PR verification log (issue #6228 / PR tasks)

## Implementation

- `MonteCarloRunner._resolve_regime_labels` maps production `RegimeSettings.risk_on_label` / `risk_off_label` to cost regimes `calm` / `stress` when those keys exist on the active `CostProcess`, without rewriting labels that already match configured cost regime keys.
- `config/scenarios/monte_carlo/cost_regime_example.yml` and preset mirror enable a deterministic regime proxy (`AssetA`, rolling return) so published example paths produce both Risk-On and Risk-Off labels.
- Integration: `test_real_runner_maps_risk_off_to_stress_costs` (seed 312, fixed calm 5bps / stress 100bps, slippage 1.0 / 2.0) checks turnover×bps×slippage reconciliation and `jobs=1` vs `jobs=2` parity.
- Unit: `test_resolve_regime_labels_maps_production_labels_to_cost_regimes` pins label mapping without a full MC run.

## Tests (2026-11-17)

```text
python3 -m pytest tests/monte_carlo/test_costs.py::test_real_runner_maps_risk_off_to_stress_costs -q  # PASS
python3 -m pytest tests/monte_carlo/test_costs.py -q  # 31 passed
python3 -m pytest tests/monte_carlo/scenario/test_cost_regime_example_load.py -q  # PASS
```

## Deliberate break → revert

Locally removed the Risk-On→calm / Risk-Off→stress branches in `_resolve_regime_labels`:

```text
python3 -m pytest tests/monte_carlo/test_costs.py::test_real_runner_maps_risk_off_to_stress_costs -q  # FAILED
# regimes stayed Risk-On / Risk-Off; {5.0, 100.0} not both present in cost_bps
```

Restored mapping (`git checkout -- src/trend_analysis/monte_carlo/runner.py`):

```text
python3 -m pytest tests/monte_carlo/test_costs.py::test_real_runner_maps_risk_off_to_stress_costs -q  # PASS
```

## PR task checklist (reconciled)

- [x] `runner.py` — regime label → calm/stress mapping when cost regimes define those keys
- [x] `cost_regime_example.yml` (+ preset) — regime detection for Risk-On / Risk-Off
- [x] `test_costs.py` + `test_cost_regime_example_load.py` — integration + load coverage
- [x] Named test `test_real_runner_maps_risk_off_to_stress_costs` passes
- [x] Deliberate break → revert recorded above
- [x] Full `tests/monte_carlo/test_costs.py` passes