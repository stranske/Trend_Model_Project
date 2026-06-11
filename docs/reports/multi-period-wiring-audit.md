# Multi-Period Wiring Audit

Issue: [#5536](https://github.com/stranske/Trend_Model_Project/issues/5536)

Generated: 2026-06-11

## Summary

This audit covers the config-derived values passed from `src/trend_analysis/multi_period/engine.py` into `_call_pipeline_with_diag` at lines 1326-1353. It cross-references the baseline catalog in `tests/baseline/catalog.yaml` and the current focused baseline result:

```text
PYTHONPATH=src python -m pytest tests/baseline/test_directional.py tests/baseline/test_wiring.py -q -rA
8 passed, 3 skipped
```

The skipped report-only entries are:

- `portfolio.weighting_scheme`: follow-up [#5537](https://github.com/stranske/Trend_Model_Project/issues/5537)
- `regime.enabled`: already covered by source issue [#5533](https://github.com/stranske/Trend_Model_Project/issues/5533) and follow-up PR [#5535](https://github.com/stranske/Trend_Model_Project/pull/5535)
- `portfolio.robustness.shrinkage.enabled`: follow-up [#5538](https://github.com/stranske/Trend_Model_Project/issues/5538)

## Classification Table

| Parameter or value | Classification | Evidence | Notes |
|---|---|---|---|
| `vol_adjust.target_vol` | confirmed-wired | `engine.py:1332`; `tests/baseline/catalog.yaml:25-34`; `tests/baseline/catalog.yaml:116-119`; `pipeline_helpers.py:78-91`; focused baseline passed `vol_target_up` and `vol_adjust_toggle`. | The directional catalog entry remains `enforce: false` because interactions with floor/leverage can be model-specific, but the current demo fixture moves output and the toggle is enforced. |
| `run.monthly_cost` / resolved monthly cost | confirmed-wired | `engine.py:1333`; `engine.py:1949`; `src/trend_analysis/multi_period/engine.py:331-348`; `src/trend_analysis/stages/portfolio.py:571-572`. | Resolved before the call site and applied to in/out return frames. Not a baseline priority parameter. |
| `vol_adjust.floor_vol` | confirmed-wired | `engine.py:1334`; `src/trend_analysis/pipeline.py:230-283`; `src/trend_analysis/pipeline_runner.py:181-220`; `src/trend_analysis/stages/preprocessing.py:75-130`. | Forwarded through the pipeline wrapper and consumed by preprocessing as the realized-volatility floor. |
| `vol_adjust.warmup_periods` | confirmed-wired | `engine.py:1335`; `src/trend_analysis/pipeline.py:230-283`; `src/trend_analysis/pipeline_runner.py:181-220`; `src/trend_analysis/stages/preprocessing.py:75-130`. | Forwarded through the pipeline wrapper and consumed by preprocessing as the warmup-period count. |
| `portfolio.selection_mode` | confirmed-wired | `engine.py:1336`; `engine.py:2385-2394`; `tests/baseline/README.md:67-70`; `tests/baseline/test_rank_fund_count_limits.py:29-89`. | Selection-count wiring was recently fixed around `rank.n`; mode behavior has dedicated tests outside the catalog. |
| `portfolio.random_n` | confirmed-wired | `engine.py:1337`; `engine.py:1624-1629`; `pipeline_helpers.py:269-327`. | Used as the default target count when rank-specific target count is absent. |
| `portfolio.custom_weights` | confirmed-wired | `engine.py:1338`; `engine.py:4197-4215`; `src/trend_analysis/stages/portfolio.py:288-415`. | Forwarded to the pipeline and explicitly used by the manual turnover-cost rerun path. |
| `portfolio.rank` / `portfolio.rank.n` | confirmed-wired | `engine.py:1339`; `tests/baseline/catalog.yaml:47-58`; `tests/baseline/test_directional.py:23-31`; `tests/baseline/test_rank_fund_count_limits.py:29-89`. | Fixed by the #5531/#5532 chain; focused baseline passed `rank_n_down`. |
| `portfolio.manual_list` | confirmed-wired | `engine.py:1340`; `engine.py:4197-4215`; `src/trend_analysis/stages/portfolio.py:288-415`. | Manual funds are consumed by the pipeline/manual rerun path. |
| `portfolio.indices_list` | confirmed-wired | `engine.py:1341`; `src/trend_analysis/monte_carlo/config.py:89-115`. | Used for risk-free and benchmark candidate resolution. Not a baseline priority parameter. |
| `benchmarks` | confirmed-wired | `engine.py:1342`; `pipeline_helpers.py:94-120`; `engine.py:740-746`; `src/trend_analysis/monte_carlo/config.py:89-115`. | Used to resolve regime proxy and risk-free/benchmark columns. |
| `seed` | confirmed-wired | `engine.py:1343`; `src/trend_analysis/api.py:454-456`; `src/trend_analysis/pipeline_runner.py:33-53`. | Forwarded to the pipeline and used to seed deterministic runs. |
| `data.missing_policy` | confirmed-wired | `engine.py:1344`; `src/trend_analysis/data.py:279-337`; `src/trend_analysis/io/ui_ingest.py:186-235`. | Validated and passed into data loading/cleaning policy paths. |
| `data.missing_limit` | confirmed-wired | `engine.py:1345`; `src/trend_analysis/data.py:327-337`; `src/trend_analysis/config/model.py:316-326`. | Validated and applied with missing data policy. |
| `vol_adjust.window` / `risk_window` | confirmed-wired | `engine.py:1346`; `pipeline_helpers.py:391-450`; `src/trend_analysis/stages/portfolio.py:707-804`. | Used by trend signal defaults and portfolio stage diagnostics. |
| `portfolio.previous_weights` | confirmed-wired | `engine.py:1321-1323`; `engine.py:1347`; `engine.py:472-507`; `engine.py:4183-4191`. | Used for turnover-cap and turnover-penalty continuity between periods. |
| `portfolio.max_turnover` | confirmed-wired | `engine.py:1348`; `engine.py:528-568`; `engine.py:705-756`; `tests/baseline/catalog.yaml:104-113`; focused baseline passed `turnover_cap_down`. | Scalar caps are wired. Regime-specific cap behavior depends on `regime.enabled` and is covered by #5533/#5535. |
| `portfolio.constraints` | confirmed-wired | `engine.py:1349`; `src/trend_analysis/engine/optimizer.py:223-314`; `tests/baseline/catalog.yaml:36-45`; `tests/baseline/catalog.yaml:82-91`; focused baseline passed `max_weight_down` and `long_only_off`. | `long_only_off` is still report-only in catalog, but the current run moved in the expected direction. |
| `regime` / `regime.enabled` | confirmed-gap | `engine.py:1350`; `pipeline_helpers.py:94-120`; `pipeline_helpers.py:269-363`; `tests/baseline/catalog.yaml:120-127`; `tests/baseline/README.md:69-80`; focused baseline skipped `regime_toggle`. | Already tracked by #5533 and follow-up PR #5535; no new issue filed from this audit. |
| `data.risk_free_column` | confirmed-wired | `engine.py:1351`; `src/trend_analysis/util/risk_free.py:15-33`; `src/trend_analysis/monte_carlo/config.py:38-74`. | Used in risk-free column selection. |
| `data.allow_risk_free_fallback` | confirmed-wired | `engine.py:1352`; `src/trend_analysis/util/risk_free.py:15-33`; `src/trend_analysis/monte_carlo/config.py:60-74`. | Used only when no explicit risk-free column is configured. |
| `signals` / `signal_spec` | confirmed-wired | `engine.py:1353`; `pipeline_helpers.py:391-450`; `src/trend_analysis/pipeline_helpers.py:403-407`. | `None` intentionally preserves default signal-window behavior; configured signals build a `TrendSpec`. |
| `portfolio.transaction_cost_bps` | confirmed-wired | `engine.py:1949`; `src/trend_analysis/multi_period/engine.py:320-348`; `tests/baseline/catalog.yaml:60-69`; focused baseline passed `cost_up`. | The single-period API warns that this is ignored outside multi-period (`api.py:476-496`), but it is resolved into the multi-period monthly cost path. |
| `metrics.rf_rate_annual` | confirmed-wired | `src/trend_analysis/api.py:511-528`; `tests/baseline/catalog.yaml:71-80`; focused baseline passed `rf_up`. | Included because it is a baseline priority parameter, although it is not directly passed at the multi-period call site. |
| `portfolio.weighting_scheme` | confirmed-gap | `engine.py:1524-1528`; `tests/baseline/catalog.yaml:93-102`; `src/trend_analysis/pipeline_entrypoints.py:109-147`; focused baseline skipped `weighting_risk_parity` with identical `max_weight`. | Follow-up issue [#5537](https://github.com/stranske/Trend_Model_Project/issues/5537). This is not a direct keyword in the audited call site but is a catalog `enforce: false` wiring candidate in the same multi-period engine. |
| `portfolio.robustness.shrinkage.enabled` | confirmed-gap | `src/trend_analysis/weights/robust_config.py:53-56`; `engine.py:2368-2378`; `tests/baseline/catalog.yaml:128-131`; focused baseline skipped `shrinkage_toggle` with identical output. | Follow-up issue [#5538](https://github.com/stranske/Trend_Model_Project/issues/5538). This is not a direct keyword in the audited call site but is a catalog `enforce: false` wiring candidate in the same multi-period engine. |

## Follow-Up Issues Filed

- [#5537](https://github.com/stranske/Trend_Model_Project/issues/5537): investigate `portfolio.weighting_scheme` risk-parity no-op in the baseline fixture.
- [#5538](https://github.com/stranske/Trend_Model_Project/issues/5538): investigate `portfolio.robustness.shrinkage.enabled` no-op in the baseline fixture.

## Reproduction

```bash
rg "_call_pipeline_with_diag" src/trend_analysis/multi_period/engine.py -A 30
PYTHONPATH=src python -m pytest tests/baseline/test_directional.py tests/baseline/test_wiring.py -q -rA
```
