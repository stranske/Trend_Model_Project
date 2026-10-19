# Regime lookback sensitivity — source #6101

Measured 2026-10-19 at base `22d68fcbcb5ece04fa30bf80d93bb17aaea76fcd`.
The production classifier consumes lookback; no production implementation or
model defaults change in this delivery.

## Real-engine result

`tests/fixtures/regime_lookback_sensitive.csv` contains eight monthly observations.
Both runs use the same config and returns, six in-sample months, and one-month
out-of-sample periods. The test explicitly compares the full configurations
with only `regime.lookback` removed. Both windows fit the available history.

| Observation | Lookback 2 | Lookback 5 |
|---|---|---|
| June 30 in-sample regime | Risk-Off | Risk-On |
| Selected funds in first period | 2 | 3 |
| July 31 out-of-sample regime | Risk-Off | Risk-On |
| User observations in matching regime report column | 1 | 1 |
| User observations in opposite regime report column | 0 | 0 |

The two last in-sample proxy returns are negative, whereas the five-observation
compounded signal is positive. The actual multi-period engine carries these
labels through selection and performance-by-regime reporting. Existing tests
retain the independent enabled/proxy selection and real export coverage.

## Executable proof

Run from the repository root with its selected Python environment:

```sh
python -m pytest tests/test_multi_period_regime_wiring.py tests/test_multi_period_regime_exports.py -q
```

Result: **6 passed**. The deliberate control replaces the production classifier's
`window = max(int(settings.lookback), 1)` with `window = 2`. The new test fails
because the five-month run incorrectly reports Risk-Off instead of Risk-On.
After byte-exact source restoration, the complete six-test command passes.
Raw [failure](evidence/regime-lookback-6101/ignored-lookback-red.txt),
[restored result](evidence/regime-lookback-6101/restored-green.txt), and
[hash/command receipt](evidence/regime-lookback-6101/proof.json) are retained.
The mutation is proof only and is absent from the delivered diff.

## Measurement limits and baseline reconciliation

The baseline `regime_toggle` remains report-only: its current comparator measures
weights and aggregate performance, not labels or regime tables. It may report
identical portfolio outputs without proving the control inert. The dedicated
sensitive real-engine regression above enforces the label, selection and report
consequence. This does not silently turn the old demo into a directional gate.

Clamped windows, a proxy that stays on one side of the threshold, disabled
regimes, or binding portfolio constraints can legitimately preserve portfolio
outputs. Historical audit snapshots describe their original fixture and time;
C3 now names this bounded measured case rather than a universal certification.
No Monte Carlo transition model, defaults, or unrelated controls are certified.
