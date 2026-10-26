# Product contract — stranske/Trend_Model_Project
_First draft generated 2026-09-20 from the audit scorecard; the repo owns this file from now on. A PR that adds a user-facing route, command or page adds a line here. The audit's Phase 1.5 scores every line below and prints any surface not listed as UNSCORED._

## Purpose
Turn fund-return data and configuration into a ranked, weighted trend portfolio/backtest and exportable analyst report.

## Primary journey
Provide returns and configuration → run model → inspect holdings and metrics → export analyst report → verify configuration/data changes affect results.

## Core functions
| id | a user can … and sees … | entry point | probe (how to exercise it; vary these determinants) | status (evidence date) |
|---|---|---|---|---|
| C1 | an analyst can run the trend model and sees ranked holdings, weights and portfolio metrics | `trend run -c CONFIG --returns CSV`; Streamlit | run demo returns with complete config; truncate to 72 rows and diff holdings/warnings | WORKS |
| C2 | an analyst can produce deliverables and sees HTML plus CSV, JSON, XLSX and TXT describing the run | `trend report -c CONFIG -i CSV --out DIR --output FILE`; GUI export | export baseline and enabled-volatility reports; compare parameters and artifacts | WORKS |
| C3 | an analyst can change configuration or data and sees results or validation change consistently | change lookback, volatility adjustment or CSV then rerun | `tests/test_multi_period_regime_wiring.py::test_multi_period_lookback_reaches_labels_selection_and_regime_report` varies `regime.lookback` on an unclamped, sensitive proxy and inspects labels, selected funds and regime report; separately vary `vol_adjust` and returns | WORKS for bounded lookback probe (2026-10-19) |
| C4 | an analyst can enable regime analysis and sees Performance by Regime in exported reports | `trend report -c CONFIG`; report/export path with `regime.enabled: true` | `tests/test_multi_period_regime_exports.py::test_multi_period_run_populates_performance_by_regime` checks the regime table, text summary and HTML Performance by Regime section | WORKS (2026-10-25) |

## Known gaps at draft time
- The draft C3 inert-control conclusion is superseded for `regime.lookback` by the real-engine test `tests/test_multi_period_regime_wiring.py::test_multi_period_lookback_reaches_labels_selection_and_regime_report`. With six in-sample observations, changing only lookback from 2 to 5 changes Risk-Off to Risk-On, selection from two to three funds, and the first out-of-sample regime-report observation. Existing tests separately exercise enabled/proxy selection and real exported regime reporting.
- Equal portfolio outputs remain legitimate when windows clamp to the same available history, the proxy does not cross a label threshold, regimes are disabled, or selection constraints absorb the difference. A weights/aggregate-performance comparison does not establish equal regime labels or reports. Historical audit snapshots retain their observation-time scope; they do not override this current bounded evidence.
- See [configuration guidance](ModelConfiguration.md#regime-lookback-and-observable-effects) and [recorded proof](reports/regime-lookback-sensitivity.md). This does not certify every configuration control or history/model combination.
