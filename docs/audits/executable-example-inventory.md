# Public API and configuration example audit

Issue #6098. Audited source: `917d3dea8b52b1d804fc69f7fe003924147c2a81`.
The bounded seam is every Python and YAML fenced block in `docs/api.md`,
`docs/phase-3/MonteCarlo.md`, and `README.md`: **18 entries**. Shell commands,
ASCII diagrams, CSV illustrations, linked guides, archived reports and historical
settings-wiring evidence are excluded. The deliberate pseudocode block is
inventoried but is not advertised as executable Python.

## Existing corrections and ownership

- Public Python API: source #6093, [PR #6115](https://github.com/stranske/Trend_Model_Project/pull/6115),
  merged `4783d0ffc247ed2e75a52a202f16d37a8f55e2a4`. Its actual four-snippet
  execution test is reused, including real pipeline results and nonempty
  XLSX/CSV/JSON artifacts.
- F41 return-model YAML: source #6097, [PR #6117](https://github.com/stranske/Trend_Model_Project/pull/6117),
  merged `917d3dea8b52b1d804fc69f7fe003924147c2a81`. Its literal-doc model
  construction, parameter-preservation and rejection tests are retained.
- Historical settings snapshots remain under #6099. This audit neither
  reconciles them nor upgrades the historical PRODUCT_CONTRACT C3 verdict.

These links establish implementation lineage, not a new post-merge acceptance
verdict. No production correction, general documentation rewrite or new issue
filing is included here.

## Complete block inventory

`API` means `docs/api.md`; `MC` means `docs/phase-3/MonteCarlo.md`.
Line numbers identify the audited source, not a moving future revision.
Confirmed entries state the level actually exercised; constructor/configuration
success does not certify an entire Monte Carlo run.

| Entry / producer | Contract and public consumer | Evidence / disposition |
| --- | --- | --- |
| API Python 1, line 13, Core Pipeline | `config.load` → `pipeline.run(config)` → metrics DataFrame. | Confirmed execution by `tests/test_active_docs_supported_commands.py::test_api_md_python_snippets_execute`; nonempty finite required metrics. |
| API Python 2, line 26, Configuration | The preceding `Config` exposes dictionary sections `data` and `portfolio`. | Confirmed execution in the same shared snippet namespace; actual config values accessed. |
| API Python 3, line 36, Metrics | Demo periodic returns → public `sharpe_ratio`, `max_drawdown`, `annual_return`; risk-free input is per period. | Confirmed execution and finite scalar assertions in that test. |
| API Python 4, line 52, Export | Named DataFrame mapping → canonical Excel/CSV/JSON exporters. | Confirmed real, nonempty files; CSV/JSON read back. No mocked exporter. |
| MC YAML 1, line 218, stationary bootstrap | Literal flat parameters → `registry.load_scenario_from_path` → runner model construction. | Confirmed by `test_published_return_model_yaml_constructs_real_models`; block length/window retained. |
| MC YAML 2, line 243, regime bootstrap | Proxy regime parameters → same loader → fitted regime model. | Confirmed by the same test, including labels, transition and buckets. The comment listing other regime kinds is not an executed alternative. |
| MC YAML 3, line 284, curated strategies | Literal curated list → scenario coercion → actual `MonteCarloRunner._resolve_strategies`. | Confirmed by new `test_published_configuration_fragments_reach_their_consumers[curated]`; names and complete overrides retained. |
| MC YAML 4, line 315, sampled strategies | Literal distribution map → scenario → actual strategy sampler. | Confirmed by the `[sampled]` case; 100 actual variants and documented rank choices. No claim about full portfolio profitability. |
| MC YAML 5, line 384, stochastic costs | Direct regime blocks → `CostProcess.from_config` via the runner. | Canonical shape checked by existing cost test; the identical nested cost values in the complete scenario reach an actual process and sample. Calm/stress lognormal means 6/18 and sigmas 0.25/0.35, positive costs, stress slippage 1.5 and the debit relation are asserted there. Standalone cost fragment is not itself a complete scenario. |
| MC YAML 6, line 441, regime-conditional turnover | Regime-keyed `portfolio.max_turnover` fragment needs a strategy/base configuration and active regime context. | **Needs investigation**: source trace reaches the runner's turnover/config handling, but this audit does not execute a complete regime-dependent cap path. Neither presumed valid nor labeled broken. |
| MC Python 7, line 470, multi-period loop | Explicit `# Pseudocode`, undefined strategy/path context and ellipsis. | **Intentional illustration**, not runnable public API. No execution claim. |
| MC YAML 8, line 499, folds | Literal explicit dates/lookback → scenario loader → real `FoldGenerator.from_config`. | Confirmed configuration construction by new `[folds]` case. Producing those historical folds requires appropriate ten-year price history; generation/execution is unproved here. |
| MC YAML 9, line 543, scenario index | Two named relative files → registry entry/path/tag loading. | **Needs investigation** for exact snippet execution; both referenced repository scenario files exist, and `registry.list_scenarios` is the source consumer. This audit does not silently substitute another index and claim this block passed. |
| MC YAML 10, line 558, complete scenario | Entire published mapping → actual registry → fitted return model, curated plus 100 sampled strategies, cost process, disabled folds, output settings. | Confirmed by new `test_complete_published_scenario_reaches_actual_model_strategy_and_cost_consumers`. Base-config path is made absolute only to supply temporary-file context. The 2,000-path simulation and output bundle are not run. |
| README YAML 1, line 151, single-period split | Example date-window fragment → full `Config`/single-period pipeline context. | **Needs investigation** as a literal YAML fragment; the separate referenced `config/single_period.yml` and Python example execute, but have their own dates. No conflation of the two payloads. |
| README Python 2, line 169, simulation | Caller-provided returns → `api.run_simulation` → detailed score frame. | Confirmed exact snippet with `demo/demo_returns.csv` retaining its Date column; score frame shape `(5, 6)`. An initial probe incorrectly moved Date to the index and failed input validation; that rejected input is not a documentation defect or accepted result. |
| README Python 3, line 211, NAV helpers | Actual fixture bundle → `load_nav_paths` → `validate_nav_paths_df` → `make_paths`. | Confirmed exact snippet; nonempty long output shape `(72, 1)`. No new binary fixture or renderer/browser claim. |
| README YAML 4, line 270, configuration | Advertised section fragment, including `returns_file` and `weighting.method`, → canonical config contract. | **Needs investigation**: naming differs from the canonical API/guide's `csv_path` and `weighting.name`; no complete-input runtime failure is established here. Return for review before deciding whether compatibility or documentation correction is appropriate. |

## Focused recurrence gate

```sh
PYTHONPATH=src python -m pytest tests/monte_carlo/test_monte_carlo_docs_schema.py -q
PYTHONPATH=src python -m pytest tests/test_active_docs_supported_commands.py::test_api_md_python_snippets_execute -q
```

The named Monte Carlo gate passes **26 tests**, retaining all prior F41 tests
and adding three fragment-to-consumer cases plus one complete-scenario case.
The existing API gate passes **1 test** that executes all four Python blocks
and real exporters (the initial combined run: **27 passed**).

For actual registry coupling, temporarily replace the `_parse_scenario`
`"strategy_set": strategy_set` assignment with `"strategy_set": None`, then
run the new complete-scenario test. It **fails** because documented strategy
names/population no longer reach the runner. Restore the exact bytes and rerun
the named gate: **26 passed**. Registry SHA-256 before and after restoration:
`b8727ae6f0c54f2526032a85afbbf4e5ce6e4be99abc262eaf4cd66ac63d460c`.

Raw inventory, README probes, RED/GREEN output and restoration proof are committed
in [evidence/executable-examples-6098](evidence/executable-examples-6098/).
The transcript paths use `$CHECKOUT` and `$REPO_VENV` for portable provenance. Ruff and shared Black validate the changed test; the repository
venv has no Black module, so no formatter installation or dependency edit was
made. The deliverable is only this inventory and tests; registry source is
unchanged. CI, post-push review and post-merge verification remain separate.

## Findings returned for review

The four **needs-investigation** configuration entries remain an explicit queue
for review, not confirmed defects or acceptance waivers. No additional production
mismatch is claimed from a parse-only result, an incomplete fragment, an input
mistake or an unexecuted snippet. Preserve the two known source owners above;
do not automatically file follow-ups or bundle engine changes into this audit.

## Review recurrence proof

The complete-scenario test also requires the fitted SPX labels, transition and
regime buckets, and the exact published calm/stress means and sigmas. Disconnecting
the actual runner proxy argument makes that test fail; independently replacing
the actual cost-parser sigma with 0.99 also makes it fail. Both production files
were restored byte-identically, and the full named gate then passed 26 tests.
The two RED transcripts, restored GREEN and source hashes are in the same
committed evidence directory. No production change is included.
