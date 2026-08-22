# Collab feedback and legacy-removal closeout

**Status:** Complete.
**Scope:** 120 findings from both Collab-Deliverables feedback rounds plus all nine legacy-removal phases.
**Reviewed baseline:** TMP `1a557a0f` (after PR #5975); every residual found there is disposed of in the corrective PR for issue #5969.
**Orchestrator record:** review `final-two-round-feedback-closeout-20260822`, plan SHA-256 `ab9494df67908bcda5384ce844f0ea7028f8769b7e92e543c29b62cbc215a082`; 27/27 partitions and 129/129 items returned valid provenance-bearing results.

## Conclusion

The colleague's review contained substantial useful work. It correctly identified real residuals in loader and entrypoint duplication, cache-degradation observability, package namespace ownership, GUI theme behavior, shell error handling, duplicate validation, TrendSpec parsing, test-driven pipeline facades, group-cap projection, shrinkage naming, and incomplete legacy phases. Those observations are credited below and were implemented. It also contained name-scan noise and conclusions that overlooked already-effective implementations; those are retained as historical findings with the current evidence rather than being converted into unnecessary rewrites.

No compatibility surface is retained merely for a nonexistent installed user base. The few retained designs are current capabilities with distinct behavior: the metrics registry owner, optional notebook rank widgets, intentional JSON/report adapters, and distinct weighting algorithms.

## Disposition rules

- **Implemented** means the concern was useful and the current corrective PR changes code, tests, or active documentation to close it.
- **Verified satisfied** means the pre-correction review found current evidence that the concern was already effectively addressed.
- **Historical observation** means the statement was once relevant or later corrected, but is not a current defect.
- **Intentional adapter/retained** means current behavior is distinct and useful; the ledger states why it is not legacy duplication.

## Complete item ledger

### R1 PR28 (p001-round-1-pr-28)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR28-01-eight-shared-cli-helper-name-duplicates` | Verified satisfied | The cross-CLI duplicates no longer exist as a product surface. The unified implementation retains one owner per behavior, with three helper functions extracted into a shared module; the ledger's generic disposition is not sufficiently explicit for this named finding. Evidence: `src/trend/cli.py:18-30,52-60; src/trend/cli_helpers.py:9-68; src/trend/commands/report_export.py:151-511`. |
| `R1-PR28-02-load-configuration-3-copies` | Verified satisfied | The three-copy concern is satisfied through retirement of both legacy containers, not mechanical extraction. The ledger's disposition is accurate. Evidence: `src/trend/cli.py:733-751`. |
| `R1-PR28-03-json-default-4-copies` | Intentional adapter | The proposed single universal serializer was correctly not applied. The substantive duplication concern is satisfied by a sound alternative: shared conversion primitives with intentional, tested shape adapters. The ledger is stale. Evidence: `src/trend_analysis/util/json_compat.py:16-61`. |
| `R1-PR28-04-build-parser-2-copies-different-signatures` | Verified satisfied | The differently scoped parsers are no longer coexisting runtime code. The substantive concern is fully satisfied; the ledger is directionally correct but stale. Evidence: `src/trend/cli.py:141-415`. |
| `R1-PR28-05-src-trend-config-schema-py-vs-src-trend-analysis` | Intentional adapter | This is an intentional two-tier validation design with runtime composition and parity coverage, not duplicate configuration ownership. The ledger disposition is accurate. Evidence: `src/trend/config_schema.py:1-10,36-43`. |
| `R1-PR28-06-src-trend-input-validation-py-vs-src-trend-valid` | Intentional adapter | The similarly named modules enforce different data representations at different stages. Their separation is intentional and behaviorally exercised; the ledger is stale. Evidence: `src/trend/input_validation.py:1-19,215-220; src/trend/validation.py:1-14,49-77,116-138`. |

### R1 PR28 (p002-round-1-pr-28)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR28-07-src-trend-compat-entrypoints-py-deprecation-time` | Verified satisfied | The substantive concern is resolved by the stronger direct-removal design: the compatibility module and all six installed aliases are absent, while trend is the sole analysis CLI. No deprecation timeline is needed because there is no retained compatibility surface. The only remaining correction is the stale ledger disposition. Evidence: `collab-prs/pr-28/cli-duplicate-cleanup.md:138-142`. |

### R1 PR29 (p003-round-1-pr-29)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR29-01-datasettings-class-defined-in-both-trend-config-` | Verified satisfied | The duplicate public type name was removed rather than retained as an intentional tier adapter; callers now have distinguishable Tier-1 and Tier-2 types. Evidence: `src/trend/config_schema.py:36-43,56-80; src/trend_analysis/config/model.py:189-204`. |
| `R1-PR29-02-validationerror-name-collision-with-pydantic` | Verified satisfied | The confusing DTO name was removed, eliminating the package-level collision with Pydantic's ValidationError. Evidence: `src/trend_analysis/config/validation.py:24-35; src/trend_analysis/config/__init__.py:27-32,57-60`. |
| `R1-PR29-03-config-bridge-py-validates-only-tier-1-fields` | Verified satisfied | The original unchecked-field defect is effectively fixed, using the proposed Tier-2 validation plus an additional semantic-validation pass. Evidence: `src/trend_analysis/config/bridge.py:74-113`. |
| `R1-PR29-04-config-legacy-py-removal-consideration` | Implemented | `docs/ConfigMap.md`; canonical config-model loader named and deleted module remains absence-gated. |

### R1 PR33 (p004-round-1-pr-33)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR33-01-1-run-analysis-py-orphaned-cli-module-delete-can` | Verified satisfied | The orphaned single-period runner is removed from source, runtime references, installed scripts, and wheel contents. Evidence: `pyproject.toml:64-66`. |
| `R1-PR33-02-2-run-multi-analysis-py` | Verified satisfied | The orphaned multi-period runner and its lazy-loader exposure are removed, with canonical `trend` packaging verified. Evidence: `src/trend_analysis/__init__.py:104-118; pyproject.toml:64-66`. |
| `R1-PR33-03-3-typing-py` | Implemented | Deleted unused `trend_analysis.typing` contract and its test; live engine owns its structural result alias. |
| `R1-PR33-04-4-ci-fixture-files-polluting-the-trend-analysis-` | Verified satisfied | The fixtures are correctly isolated as workflow-test runtime seams and excluded from the production namespace. Evidence: `tests/workflows/fixtures/__init__.py:1-5; tests/workflows/test_autofix_samples.py:10-14; tests/workflows/test_autofix_probe_module.py:3-9; tests/workflows/test_autofix_repo_regressions.py:10`. |
| `R1-PR33-05-5-typing-py-vs-typing-py-confusing-naming` | Implemented | Deleted the unused public `typing.py`; retained the used private numerical `_typing.py` aliases. |
| `R1-PR33-06-6-cashpolicy-dual-exposure` | Implemented | The later legacy campaign correctly rejected the dual exposure: callers now import the canonical `cash_policy` module and `rebalancing` no longer re-exports `CashPolicy`. |

### R1 PR37 (p005-round-1-pr-37)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR37-01-1-pipeline-entrypoints-py-redundant-duplicate-as` | Historical observation; current concern resolved | The redundant assignment was removed as part of the shared risk-free resolution change. The finding is historical only, but the ledger has not been reconciled. Evidence: `src/trend_analysis/pipeline_entrypoints.py:288-292`. |
| `R1-PR37-02-2-pipeline-entrypoints-py-inconsistent-risk-free` | Historical observation; current concern resolved | The substantive divergence is eliminated by a sound shared design and behavioral parity tests. The dated ledger's grouped active disposition is stale for this finding. Evidence: `src/trend_analysis/pipeline_entrypoints.py:60-67,144-146,290-292`. |
| `R1-PR37-03-3-pipeline-entrypoints-py-50-lines-duplicated-be` | Implemented | `pipeline_entrypoints.py` now resolves common inputs once; parity coverage protects both result-shaping tails. |
| `R1-PR37-04-4-data-py-load-csv-and-load-parquet-share-50-lin` | Implemented | `data.py` uses a shared loader skeleton with format-specific readers and preserved exception policy. |
| `R1-PR37-05-5-walk-forward-py-vs-engine-walkforward-py-two-w` | Intentional adapter | These are intentionally separate layers, not competing implementations. Their module docstrings and distinct callers make the boundary adequate; forcing delegation would distort the engine's aggregate-only contract. The ledger disposition is accurate. Evidence: `src/trend_analysis/walk_forward.py:1-24,268-379,434-467`. |
| `R1-PR37-06-6-presets-py-and-signal-presets-py-two-parallel-` | Intentional adapter | The original hardcoded parallel registry has been replaced by an intentional adapter over the canonical YAML registry. No preset consolidation code remains necessary; the ledger is stale. Evidence: `src/trend_analysis/signal_presets.py:1-6,61-96,117-130`. |

### R1 PR38 (p006-round-1-pr-38)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR38-01-1-trend-model-cli-py-imports-8-private-symbols-f` | Verified satisfied | The private-import risk cannot recur because the consuming legacy CLI and package are removed, including from built wheels. The dated ledger disposition is incomplete. Evidence: `1a557a0f:tests/test_legacy_surface_absence.py:126-138,383-387; isolated pytest selected tests exit 0`. |
| `R1-PR38-02-2-src-cli-py-dev-only-utility-not-wired-as-an-en` | Verified satisfied | The unwired utility is gone, while supported report behavior is provided by trend. The residual src namespace is a separate current developer-path dependency, not a live old CLI. Evidence: `1a557a0f:pyproject.toml:64-81; tests/test_cli_installed.py:13-68; isolated pytest selected tests exit 0`. |
| `R1-PR38-03-3-trend-portfolio-app-init-py-empty-compat-stub-` | Verified satisfied | The obsolete package is removed from checkout and wheel output; the old concern is fully addressed. The dated ledger is stale. Evidence: `1a557a0f:tests/test_legacy_surface_absence.py:126-138,383-387; isolated pytest selected tests exit 0`. |
| `R1-PR38-04-4-health-summarize-init-py-private-helpers-expor` | Verified satisfied | The finding concerns a test-only source fallback, not the live CI script. Its public export boundary now correctly excludes private helpers, but the ledger remains stale. Evidence: `1a557a0f:src/health_summarize/__init__.py:393-397`. |
| `R1-PR38-05-5-trend-model-spec-py-config-access-helpers-dupl` | Intentional adapter | The legacy package was removed, but equivalent names remain in two deliberately different configuration layers. The original low-risk duplication concern is resolved by a sound differentiated design; the ledger is stale. Evidence: `1a557a0f:src/trend/spec.py:92-118; 1a557a0f:src/trend_analysis/pipeline_helpers.py:45-75`. |

### R1 PR39 (p007-round-1-pr-39)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR39-01-1-nav-path-required-charts-defined-twice` | Verified satisfied | The duplicate-definition defect is historical: current MC validation and public export use one canonical constant, and the ledger's explicit completed disposition remains accurate. Evidence: `src/trend/mc/viz.py:14`. |
| `R1-PR39-02-2-init-matplotlib-copied-between-the-two-reporti` | Verified satisfied | The copied setup is eliminated through the proposed shared-module design; the completed ledger disposition is accurate. Evidence: `src/trend/reporting/_matplotlib.py:8-24`. |
| `R1-PR39-03-3-trend-cli-py-at-2-326-lines-carries-too-many-d` | Verified satisfied | The substantive monolith concern is addressed by a sound modular dispatcher design. The remaining dispatcher code is normal command integration, not evidence that the original 2,326-line multi-owner defect persists; the dated ledger is stale. Evidence: `src/trend/cli.py:18-60; src/trend/cli.py:1015-1098`. |
| `R1-PR39-04-4-the-legacy-cli-shim-in-trend-cli-py-is-non-obv` | Verified satisfied | This was a test-only runtime compatibility seam with no independent product value. It has been removed, so the readability concern no longer applies; the ledger has not caught up. Evidence: `src/trend_analysis/cli.py absent at 1a557a0fd3816257f9ac8549f23797dddee3cfbe; pyproject.toml:64-66`. |

### R1 PR41 (p008-round-1-pr-41)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR41-01-1-duplicate-timed-stage-implementation-in-signal` | Intentional adapter | The helpers share stopwatch mechanics but serve intentionally different logging contracts, so consolidation would weaken the signal-specific diagnostic seam. Evidence: `src/trend_analysis/signals.py:96-106; src/trend_analysis/perf/timing.py:34-78`. |
| `R1-PR41-02-2-metrics-init-py-is-monkey-patching-builtins` | Verified satisfied | The import-time builtins mutation and its unsupported compatibility surface have been removed. Evidence: `src/trend_analysis/metrics/__init__.py:442-456`. |
| `R1-PR41-03-3-synthetic-tests-legacy-metrics-module-collides` | Verified satisfied | Both the synthetic runtime oracle and the collision-prone legacy product surface are gone. Evidence: `src/trend_analysis/metrics/__init__.py:399-456`. |
| `R1-PR41-04-4-metrics-init-py-is-doing-too-many-things` | Retained intentionally | The initializer is the one public registry and import surface; side effects and compatibility bindings are gone, and no behavioral duplication remains. Splitting it would add ownership seams without deleting a product or compatibility surface. |
| `R1-PR41-05-5-hrp-s-catch-all-except-exception` | Intentional adapter | The catch-all is now an observable, diagnosed safety fallback rather than a silent bug sink; the requested traceback visibility is effective. Evidence: `src/trend_analysis/weights/hierarchical_risk_parity.py:158-170`. |
| `R1-PR41-06-6-robustriskparity-diagnostic-mismatch` | Historical observation; current concern resolved | The reported diagnostic mismatch is historical only; current behavior accurately reports loading regardless of its trigger. Evidence: `src/trend_analysis/weights/robust_weighting.py:342-366,377-415`. |

### R1 PR41 (p009-round-1-pr-41)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR41-07-7-erc-s-regularisation-thresholds-are-hard-coded` | Historical observation; current concern resolved | The original hard-coded-threshold finding is historical. A typed policy is a sound alternative to mirroring robust-engine scalar constructor parameters, and it directly controls ERC's numerical operations. The dated ledger is incomplete because it predates and does not record this implementation. Evidence: `src/trend_analysis/weights/equal_risk_contribution.py:15-72`. |
| `R1-PR41-08-8-weights-init-py-underexports` | Historical observation; current concern resolved | The claimed absence of a canonical public-surface line is historical. Current __all__ plus its contract test intentionally define a narrow surface; exporting every internally used helper would widen the API without demonstrated product value. The ledger does not state this explicit disposition. Evidence: `src/trend_analysis/weights/__init__.py:1-15`. |
| `R1-PR41-09-9-rolling-cache-silently-disables-itself` | Implemented | `perf/rolling_cache.py` emits one path-free warning for temporary or disabled storage; forced-failure tests cover both paths. |

### R1 PR42 (p010-round-1-pr-42)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR42-01-1-sortino-in-harness-compute-metrics-is-missing-` | Historical observation; current concern resolved | The reported deannualization defect is historical only; the current delegation and value regression are sound, and the ledger is accurate. Evidence: `src/trend_analysis/backtesting/harness.py:598-612`. |
| `R1-PR42-02-2-util-missing-py-coerce-policy-silently-aliases` | Historical observation; current concern resolved | The silent alias is gone; remaining bfill strings in mocked tests are not production policy behavior. The ledger disposition is accurate. Evidence: `src/trend_analysis/util/missing.py:73-77`. |
| `R1-PR42-03-3-two-apply-missing-policy-functions-with-differ` | Verified satisfied | The incompatible market_data implementation has been removed and all current IO validation uses the canonical missing-policy function. The ledger is inaccurate. Evidence: `src/trend_analysis/io/market_data.py:27-32; src/trend_analysis/io/market_data.py:593-600`. |
| `R1-PR42-04-4-infer-periods-per-year-is-duplicated-between-b` | Intentional adapter | Frequency inference is fully consolidated. The retained private import aliases are intentional references to one implementation, not duplicate logic; the ledger is stale. Evidence: `src/trend_analysis/backtesting/harness.py:16-22; src/trend_analysis/engine/walkforward.py:11-14; src/trend_analysis/util/frequency.py:159-192`. |
| `R1-PR42-05-5-date-fix-logic-is-duplicated-between-io-market` | Intentional adapter | The divergent market_data repair surface was removed. Current adapters delegate to one date-correction engine and the UI avoids a second pass; the ledger is stale. Evidence: `src/trend_analysis/io/market_data.py:23-32; src/trend_analysis/io/market_data.py:350-385`. |
| `R1-PR42-06-6-read-uploaded-file-in-validators-py-has-copy-p` | Implemented | The follow-up legacy review correctly found no product owner for the remaining module; `io.validators` and its test-only callers are deleted, while canonical upload and market-data validation remain covered. |

### R1 PR43 (p011-round-1-pr-43)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR43-01-1-direct-key-access-on-result-dicts-in-two-expor` | Verified satisfied | The original KeyError concern was real, but current code resolves it with a sounder explicit contract. Only the corrective ledger remains inaccurate. Evidence: `src/trend_analysis/export/__init__.py:44-80,485,777`. |
| `R1-PR43-02-2-two-openpyxl-proxy-pairs-doing-the-same-job-in` | Intentional adapter | The apparent pair is an intentional single-implementation adapter arrangement for distinct workbook entry forms. The substantive duplication is resolved and the ledger is accurate. Evidence: `src/trend_analysis/export/__init__.py:146-255`. |
| `R1-PR43-03-3-portfolio-series-defined-three-times-in-export` | Verified satisfied | The duplicated implementation is removed and the shared helper preserves the required weighting behavior. The ledger disposition is accurate. Evidence: `src/trend_analysis/export/__init__.py:28,595-596,857-858,1333-1334`. |
| `R1-PR43-04-4-git-hash-duplicated-in-export-bundle-py-and-re` | Verified satisfied | The duplicated and overly broad implementation is gone. Remaining private names are test seams around one shared implementation, not a second production helper. Evidence: `src/trend_analysis/util/git.py:9-20; src/trend_analysis/export/bundle.py:15-25; src/trend_analysis/reporting/run_artifacts.py:15-28`. |
| `R1-PR43-05-5-to-nav-wide-duplicated-in-charts-rolling-panel` | Historical observation; current concern resolved | The duplicate-helper assertion is historical. Current charts deliberately use the stricter canonical adapter, so no duplicate production surface remains and the ledger is accurate. Evidence: `src/trend_analysis/viz/charts/rolling_panel.py:10,42-47; src/trend_analysis/viz/charts/seasonality_heatmap.py:11,35-40; src/trend_analysis/viz/adapters.py:467-497`. |

### R1 PR44 (p012-round-1-pr-44)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR44-01-1-bare-non-package-qualified-import-in-llm-schem` | Implemented | Moved generic top-level `src/utils` into `trend_analysis.util.paths`, migrated callers, narrowed package discovery, and added isolated-wheel coverage. |
| `R1-PR44-02-2-detect-result-hallucinations-silences-the-call` | Historical observation; current concern resolved | The original finding conflated complete validation logging with the narrower hallucination-reporting API. Current behavior avoids duplicate logs while preserving all hallucination findings. Evidence: `src/trend_analysis/llm/result_validation.py:203-282`. |
| `R1-PR44-03-3-configpatchchain-run-and-configpatchvariantsch` | Verified satisfied | The duplication has been removed by the recommended kind of shared execution scaffold, with explicit variant-specific hooks retained only where behavior differs. Evidence: `src/trend_analysis/llm/chain.py:417-746`. |
| `R1-PR44-04-4-resultsummarychain-reimplements-bind-llm-rathe` | Verified satisfied | The substantive drift risk is removed without forcing incompatible inheritance: both chain families share the runtime behavior through a dedicated mixin. Evidence: `src/trend_analysis/llm/chain.py:147-236; src/trend_analysis/llm/chain.py:755-787`. |
| `R1-PR44-05-5-read-env-float-defined-identically-in-two-file` | Verified satisfied | The duplicate helper is eliminated through a single shared implementation; its stricter invalid-value behavior is consistent across all current consumers. Evidence: `src/trend_analysis/llm/_env.py:1-15; src/trend_analysis/llm/chain.py:23; src/trend_analysis/llm/result_feedback.py:10`. |

### R1 PR47 (p013-round-1-pr-47)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR47-01-1-css-variable-name-has-a-leading-space-theme-sw` | Implemented | GUI theme selection now updates a consumed DOM attribute/style; regression coverage proves the selected value is applied. |
| `R1-PR47-02-2-ui-rank-widgets-py-imports-ipywidgets-uncondit` | Retained intentionally | Notebook rank widgets remain a distinct optional UI from Streamlit; `ipywidgets` is lazy-loaded and headless import behavior is tested. |
| `R1-PR47-03-3-asyncio-get-event-loop-call-later-is-deprecate` | Historical observation; current concern resolved | The deprecated/undriven-loop failure mode is removed by a sound running-loop-or-immediate-clear design. The ledger is stale. Evidence: `src/trend_analysis/gui/app.py:429-439`. |
| `R1-PR47-04-4-risky-change-guard-middleware-mutates-the-priv` | Historical observation; current concern resolved | The private Starlette mutation is gone and the supported receive-replay wrapper preserves downstream parsing behavior. The ledger disposition is incomplete. Evidence: `src/trend_analysis/api_server/__init__.py:77-83,99-134`. |
| `R1-PR47-05-5-multiple-default-bindings-to-0-0-0-0` | Historical observation; current concern resolved | The three named runtime defaults are effectively fixed. Explicit external deployment binds are intentional; only the documentation is stale. Evidence: `src/trend_analysis/api_server/__main__.py:3-7; src/trend_analysis/api_server/__init__.py:192-211; src/trend_analysis/proxy/cli.py:28-34; src/trend_analysis/proxy/server.py:237-258`. |

### R1 PR48 (p014-round-1-pr-48)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR48-01-1-coverage-guard-main-returns-exit-code-0-on-eve` | Historical observation; current concern resolved | The return-0 defect is fixed; the ledger's explicit completed disposition is accurate. Evidence: `tools/coverage_guard.py:717-752`. |
| `R1-PR48-02-2-coverage-guard-py-uses-deprecated-datetime-utc` | Historical observation; current concern resolved | The deprecated UTC call is absent from current coverage_guard.py, and the ledger is accurate. Evidence: `tools/coverage_guard.py:10-17,193-243`. |
| `R1-PR48-03-3-llm-provider-setup-langsmith-tracing-overwrite` | Historical observation; current concern resolved | The explicit-opt-out overwrite concern is fixed. The ledger is stale; import-time setup remains, but it no longer overrides user preference. Evidence: `tools/llm_provider.py:54-81`. |
| `R1-PR48-04-4-langchain-client-is-reasoning-model-uses-impor` | Historical observation; current concern resolved | The inline dynamic import is removed and behavior remains correct; the ledger disposition is stale. Evidence: `tools/langchain_client.py:10-14,143-154`. |
| `R1-PR48-05-5-post-ci-summary-collect-required-segments-has-` | Historical observation; current concern resolved | The redundant local import is absent and the ledger's completed disposition is accurate. Evidence: `tools/post_ci_summary.py:10-17,651-712`. |

### R1 PR49 (p015-round-1-pr-49)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR49-01-1-check-branch-sh-line-222-duplicate-then-makes-` | Historical observation; current concern resolved | The duplicate-then defect is historical only. Current syntax and the ledger's completed disposition agree. Evidence: `scripts/check_branch.sh:216-225`. |
| `R1-PR49-02-2-test-release-sh-line-36-sed-i-bak-is-gnu-bsd-i` | Historical observation; current concern resolved | The GNU/BSD sed defect is historical only. The current implementation uses the sound portable form and the ledger is accurate. Evidence: `scripts/test-release.sh:34-37`. |
| `R1-PR49-03-3-quick-check-sh-lines-22-26-after-a-command-sub` | Implemented | `quick_check.sh` handles `git diff` failure inside an `if` guard under `set -e`; a runtime failure-path test protects the warning/continuation behavior. |
| `R1-PR49-04-4-open-pr-from-issue-sh-lines-53-and-59-escaped-` | Historical observation; current concern resolved | The escaped-backtick defect is historical only. Current output construction is correct and the ledger is accurate. Evidence: `scripts/open_pr_from_issue.sh:52-60`. |
| `R1-PR49-05-5-validate-llm-deps-py-lines-40-44-exact-pin-maj` | Historical observation; current concern resolved | The hardcoded exact-series duplication is historical only: current code uses the proposed metadata-sourced range design. The ledger's active disposition is stale, although intentional upper bounds still require metadata updates when compatibility is deliberately expanded. Evidence: `scripts/validate_llm_deps.py:38-49,92-129`. |

### R1 PR50 (p016-round-1-pr-50)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR50-01-1-ci-cosmetic-repair-py-line-307-deprecated-date` | Historical observation; current concern resolved | The cited defect was real in the reviewed revision but is resolved at the current head. The ledger is stale. Evidence: `1a557a0fd3816257f9ac8549f23797dddee3cfbe:scripts/ci_cosmetic_repair.py:30,300-308`. |
| `R1-PR50-02-2-run-multi-demo-py-module-level-side-effects-ma` | Intentional adapter | The private broad-validation runner remains intentionally executable only through public main(); importing scripts.run_multi_demo no longer executes it. This is a sound alternative to in-place wrapping of the former monolith, but the ledger is stale. Evidence: `1a557a0fd3816257f9ac8549f23797dddee3cfbe:scripts/run_multi_demo.py:1-22`. |
| `R1-PR50-03-3-compare-perf-py-default-threshold-reads-env-as` | Historical observation; current concern resolved | The whole-file float conversion has been replaced by a sound named-key dotenv design. The ledger is stale. Evidence: `1a557a0fd3816257f9ac8549f23797dddee3cfbe:scripts/compare_perf.py:41-76`. |
| `R1-PR50-04-4-evaluate-settings-effectiveness-py-lines-141-1` | Historical observation; current concern resolved | This is a historical defect with a correct current implementation and accurate ledger disposition. Evidence: `1a557a0fd3816257f9ac8549f23797dddee3cfbe:scripts/evaluate_settings_effectiveness.py:141-144`. |
| `R1-PR50-05-5-run-real-model-py-line-121-rf-series-loc-portf` | Historical observation; current concern resolved | The current solution is stronger than the proposed bare reindex because it also defines safe missing-date behavior. Evidence: `1a557a0fd3816257f9ac8549f23797dddee3cfbe:scripts/run_real_model.py:30-38,123-136`. |

### R1 PR51 (p017-round-1-pr-51)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR51-01-1-pr-verifier-py-line-24-from-scripts-import-api` | Historical observation; current concern resolved | The cited missing-module failure was real in the archived review but is resolved in current TMP; the ledger's disposition is accurate. Evidence: `scripts/langchain/pr_verifier.py:27,652; scripts/api_client.py:255-280`. |
| `R1-PR51-02-2-injection-guard-py-regex-patterns-don-t-cross-` | Historical observation; current concern resolved | The newline bypass no longer reproduces; the ledger accurately records this finding as fixed. Evidence: `scripts/langchain/injection_guard.py:85,88-156`. |
| `R1-PR51-03-3-label-matcher-py-token-matches-keyword-prefix-` | Historical observation; current concern resolved | The false-positive behavior is resolved by a sound explicit-alias design, but the master ledger is stale for this subfinding. Evidence: `scripts/langchain/label_matcher.py:165-190,354-359`. |
| `R1-PR51-04-4-progress-reviewer-py-build-review-payload-dead` | Historical observation; current concern resolved | The dead conditional was removed without changing the intended payload semantics; the ledger is stale for this subfinding. Evidence: `scripts/langchain/progress_reviewer.py:127-140`. |
| `R1-PR51-05-5-structured-output-py-hard-cap-of-1-repair-atte` | Historical observation; current concern resolved | The silent-truncation defect is resolved by the feedback-approved explicit-rejection design; the master ledger has not been updated accordingly. Evidence: `scripts/langchain/structured_output.py:34-35,103-113,192-223`. |

### R1 PR52 (p018-round-1-pr-52)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR52-01-1-deprecation-shims-are-still-imported-by-the-ap` | Verified satisfied | The deprecated Streamlit shim product surfaces were removed, and former app consumers now import canonical implementations. The ledger disposition is accurate. Evidence: `streamlit_app/components/guardrails.py:20-23; streamlit_app/components/csv_validation.py:14-21; streamlit_app/pages/1_Data.py:44-49`. |
| `R1-PR52-02-2-data-cache-py-hash-dataframe-returns-the-whole` | Verified satisfied | Confirmed historical defect is fixed with a stronger schema-aware digest; the master ledger is incomplete. Evidence: `streamlit_app/components/data_cache.py:53-68`. |
| `R1-PR52-03-3-csv-validation-py-line-284-logger-exception-on` | Verified satisfied | Confirmed historical defect is fixed; expected validation failures are warning-level and no longer emit tracebacks. The master ledger is incomplete. Evidence: `streamlit_app/components/csv_validation.py:282-285`. |
| `R1-PR52-04-4-state-py-values-equal-cross-type-numeric-value` | Verified satisfied | Confirmed historical defect is effectively fixed, including preservation of meaningful nonnumeric type differences; the ledger is accurate. Evidence: `streamlit_app/state.py:316-336`. |

### R1 PR53 (p019-round-1-pr-53)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R1-PR53-01-1-8-validation-py-893-if-name-main-guard-means-t` | Verified satisfied | The default product page was removed and the retained developer tool has a correct page-configuration and rendering design. The behavioral finding is satisfied; only an active documentation path is stale. Evidence: `da31d5f7; streamlit_app/developer_settings_validation.py:23-40,918-928`. |
| `R1-PR53-02-2-8-validation-py-543-reads-session-key-app-data` | Verified satisfied | The removed default surface no longer has the invalid session-key dependency, and its developer replacement uses the canonical state API. The ledger disposition is accurate. Evidence: `streamlit_app/developer_settings_validation.py:569-573; streamlit_app/state.py:70-86`. |
| `R1-PR53-03-3-8-validation-py-498-calls-private-execute-anal` | Verified satisfied | The retained developer validation capability uses the public analysis API and has a non-vacuous regression test. The ledger disposition is accurate. Evidence: `streamlit_app/developer_settings_validation.py:504-520; streamlit_app/components/analysis_runner.py:601-618`. |
| `R1-PR53-04-4-3-results-py-1370-dataframe-applymap-is-deprec` | Verified satisfied | The deprecated call was replaced by the sound current pandas API with behavioral regression coverage; the ledger disposition is accurate. Evidence: `streamlit_app/pages/3_Results.py:1451-1464,1547-1548`. |
| `R1-PR53-05-5-1-data-py-735-750-internal-performance-telemet` | Verified satisfied | The telemetry defect is fixed through an explicit developer-only gate, retaining diagnostics without exposing them to default users; the ledger disposition is accurate. Evidence: `streamlit_app/pages/1_Data.py:851-878,909-955`. |

### R2 PR61 (p020-round-2-pr-61)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR61-01-deduplicate-8-shared-cli-functions-partial` | Verified satisfied | The original five-copy claim is no longer a live duplication problem; removal of the duplicate CLI is the stronger design. Evidence: `tests/test_legacy_surface_absence.py:126-139; tests/test_legacy_surface_absence.py:383-400`. |
| `R2-PR61-02-extract-json-default-4-copies-1-not-applied` | Intentional adapter | The four names remain by design, but duplicate primitive logic has been removed without collapsing materially different artifact contracts. Evidence: `src/trend_analysis/util/json_compat.py:16-61`. |
| `R2-PR61-03-extract-load-configuration-3-copies-1-not-applie` | Verified satisfied | Signature drift among three loaders is eliminated; the former unique behavior survives under a canonical module rather than through mechanical helper consolidation. Evidence: `src/trend/cli.py:733-751`. |
| `R2-PR61-04-evaluate-retiring-trend-model-not-applied` | Verified satisfied | trend_model has been fully retired without dropping its retained run-spec capability. Evidence: `tests/test_legacy_surface_absence.py:126-139,383-400`. |
| `R2-PR61-05-document-deprecation-timeline-not-applied` | Verified satisfied | A timeline is no longer applicable: the deprecated commands and warning shim have been removed, which is stronger than documenting a future removal date. Evidence: `tests/test_legacy_surface_absence.py:67-77,126-139,415-427`. |

### R2 PR62 (p021-round-2-pr-62)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR62-01-rename-datasettings-coredatasettings-not-applied` | Verified satisfied | The collision is removed by the sounder role-specific name ResolvedDataSettings, rather than the proposed CoreDataSettings. Evidence: `src/trend/config_schema.py:36-43,56-80`. |
| `R2-PR62-02-rename-validationerror-configissue-not-applied` | Verified satisfied | The local Pydantic-name shadowing concern is resolved directly. Evidence: `src/trend_analysis/config/validation.py:24-35; src/trend_analysis/config/__init__.py:24-29,53-59`. |
| `R2-PR62-03-remove-duplicated-required-field-checks-not-appl` | Implemented | Schema validation is the sole owner of required `portfolio` and `cost_model` diagnostics; exact-once regressions cover both. |
| `R2-PR62-04-remove-builtins-trend-config-class-caching-not-a` | Verified satisfied | Interpreter-global class caching is gone; current regression coverage verifies the relevant non-mutation behavior. Evidence: `src/trend_analysis/config/models.py:271-283,545-558; commit b09c29d9`. |
| `R2-PR62-05-expand-bridge-validate-payload-to-tier-2-applied` | Intentional adapter | The bridge is an intentional two-tier adapter and substantively closes the unchecked Tier-2 field gap. Evidence: `src/trend_analysis/config/bridge.py:74-113,128-137`. |
| `R2-PR62-06-document-vol-target-risk-threshold-not-applied` | Verified satisfied | The concern is resolved by a documented, tested typed policy rather than a comment beside a magic constant. Evidence: `src/trend_analysis/config/patch.py:43-65,500-505`. |

### R2 PR62 (p022-round-2-pr-62)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR62-07-document-silent-coercion-in-build-config-from-ui` | Intentional adapter | The substantive concern is addressed by an explicit module contract, which is a sound alternative to a function-local docstring. The old absence claim is historical; the ledger is now too broad and stale for this individual finding. Evidence: `src/trend_analysis/config/ui_mapping.py:1-16,37-50`. |
| `R2-PR62-08-move-configcoveragetracker-activation-out-of-pro` | Intentional adapter | The tracker remains a production-resident diagnostic, but it is not implicitly or always activated. The explicit user-requested CLI mode is a sound intentional design, and the ledger's opt-in-coverage disposition is accurate. Evidence: `src/trend/cli.py:184-187,236-239,254-257,769-772`. |
| `R2-PR62-09-extra-migrate-off-legacy-config-not-applied` | Verified satisfied | legacy.Config is a removed product surface, not an active adapter: its module is deleted and its two cited production callers use the canonical model with behavioral parity coverage. The ledger's active disposition is stale. Evidence: `commit 6a2b264616b5a21a6975e7847cb9cd9fed9fd2a7; src/trend_analysis/config/ui_mapping.py:24,313-321; streamlit_app/components/analysis_runner.py:14-17,352-362`. |

### R2 PR63 (p023-round-2-pr-63)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR63-01-regimes-py-model-not-validated-in-normalise-sett` | Intentional adapter | Deferring model validation is a sound plugin-extensibility adapter, not a deep opaque failure: the live boundary reports the unknown name and available models. Evidence: `src/trend_analysis/regimes.py:60-67,317-329`. |
| `R2-PR63-02-regimes-py-volatility-threshold-band-annualisati` | Verified satisfied | The behavior is deliberate and now has an explicit, internally consistent unit contract plus behavioral protection; the ledger status is stale. Evidence: `src/trend_analysis/regimes.py:224-242`. |
| `R2-PR63-03-data-py-date-column-canonicalisation-is-csv-only` | Verified satisfied | The original cross-loader defect is fixed through the recommended shared canonicalisation design; the ledger has not caught up. Evidence: `src/trend_analysis/data.py:364-375,395-430,460-492`. |
| `R2-PR63-04-cross-module-frequency-normalisation-logic-proli` | Historical observation; current concern resolved | The name-level multiplicity remains, but current implementations serve distinct contracts, so it is not evidence of live duplicate normalization logic. Evidence: `src/trend_analysis/schedules.py:54-67,186-209`. |
| `R2-PR63-05-time-utils-py-month-end-boundary-at-midnight-low` | Historical observation; current concern resolved | The hypothesized intraday exclusion is real at the helper level but not a defect in the current supported monthly product surface. Evidence: `src/trend_analysis/time_utils.py:21-45; src/trend_analysis/stages/preprocessing.py:333-363`. |

### R2 PR64 (p024-round-2-pr-64)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR64-01-build-trend-spec-three-implementations-medium` | Implemented | Preset construction delegates to the canonical mapping parser; a parity test covers the intentional preset-only window clamp. |
| `R2-PR64-02-json-default-copy-in-rank-selection-py-69-medium` | Intentional adapter | The substantive concern is resolved by a sound alternative: common primitive conversion is centralized, while artifact-specific adapters intentionally preserve distinct schemas. The ledger is stale. Evidence: `src/trend_analysis/core/rank_selection.py:29-31; src/trend_analysis/core/rank_selection.py:68-78`. |
| `R2-PR64-03-two-metric-caches-low-confirm-complementary` | Intentional adapter | The caches are complementary, not duplicate implementations. The ledger conclusion is accurate but insufficiently item-specific. Evidence: `src/trend_analysis/core/rank_selection.py:138-208`. |
| `R2-PR64-04-dynamic-binding-shims-defeat-typing-low` | Implemented | Removed pipeline stage synchronization and `Any` passthrough facades; production callers and tests use canonical helpers, runner, and stages directly. The private diagnostics runner is no longer re-exported from `pipeline`. |
| `R2-PR64-05-vol-targeting-scale-appears-double-applied-to-th` | Verified satisfied | The high-severity cross-stage defect is fixed: user and equal-weight portfolio calculations consume raw returns, while constrained weights retain the intended volatility tilt. The ledger is accurate. Evidence: `src/trend_analysis/stages/portfolio.py:642-650`. |
| `R2-PR64-06-compute-stats-ignores-window-periods-per-year-me` | Verified satisfied | Annualization is now consistently explicit and propagated. The ledger is stale, not the implementation. Evidence: `src/trend_analysis/stages/portfolio.py:216-240`. |

### R2 PR65 (p025-round-2-pr-65)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `R2-PR65-01-compute-constrained-weights-sound` | Historical observation; current concern resolved | The original sound conclusion was historically inaccurate on downstream scaling, but the current one-owner allocation design and regression effectively resolve that concern; the ledger accurately records the correction. Evidence: `evidence/2026-08-12-collab-review-corrective-work.md:186-192,421`. |
| `R2-PR65-02-engine-optimizer-apply-constraints-sound` | Implemented | Group-cap redistribution now uses asset and group headroom, preserves every configured cap, and rejects jointly infeasible constraints. |
| `R2-PR65-03-three-risk-parity-family-engines-low-observation` | Retained separately | Package and class documentation distinguishes inverse-volatility, hierarchical clustering, equal-risk-contribution, and diagnostic/repair behavior; their algorithms are not copies. |
| `R2-PR65-04-robust-weighting-shrinkage-safe-mode-sound-scan` | Implemented with corrected contract | Renamed the covariance-only formulas to `matrix_diagonal` and `matrix_trace`, documented them as heuristics, updated config/UI/schema, and reject the misleading retired estimator names. |

### Legacy campaign (p026-legacy-removal-campaign)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `LEGACY-PHASE-0` | Implemented | Active structural docs now name supported modules and public calls; semantic documentation gates reject deleted runner files and private runner symbols. |
| `LEGACY-PHASE-1` | Verified satisfied | Current UI construction uses the canonical Config path and removed configuration shapes are rejection boundaries, not retained compatibility adapters. Evidence: `src/trend_analysis/config/ui_mapping.py:24; streamlit_app/components/analysis_runner.py:14-17,352-361; src/trend_analysis/config/legacy.py absent`. |
| `LEGACY-PHASE-2` | Verified satisfied | The canonical CLI contains the MC command tree and no production dependency on the retired trend_analysis.cli surface was found. Evidence: `src/trend/cli.py:18-25,61-69,134-143,259-260; src/trend/cli_commands.py:1-6`. |
| `LEGACY-PHASE-3` | Implemented | Installed entry points remain only `trend` and `trend-llm-proxy`; active-doc command-token gates reject the removed executable alias while allowing package/image names. |
| `LEGACY-PHASE-4` | Verified satisfied | The live run-spec contract is relocated to trend.spec and no active trend_model product surface remains. Evidence: `src/trend/spec.py; tests/test_spec_loader.py:10-24; src/trend_model absent`. |
| `LEGACY-PHASE-5` | Implemented | Deleted duplicate `scripts/trend-reproducible`; `scripts/trend` owns hash-seed behavior; structural index uses current pages and non-stale inventories; absence/page gates protect the result. |

### Legacy campaign (p027-legacy-removal-campaign)

| Item | Current disposition | Evidence and rationale |
| --- | --- | --- |
| `LEGACY-PHASE-6` | Implemented | Consumers import `CashPolicy` from its canonical module, rebalancing no longer re-exports it, and the test-only `io.validators` module and obsolete tests are deleted. |
| `LEGACY-PHASE-7` | Implemented | Deleted the multi-period forwarding seam; callers use the diagnostics runner directly and the absence gate forbids restoration in both pipeline modules. |
| `LEGACY-PHASE-8` | Implemented and final-gated | Unified absence, wheel, installed CLI, Streamlit, export, config, MC, demo, lint/type, and full-suite verification recorded in this document's validation section. |

## Legacy-removal campaign completion

| Phase | Exit condition | Current proof |
| --- | --- | --- |
| 0 | Inventory and replacement contracts | 129-item ledger, active-doc semantic gates, supported CLI/API references |
| 1 | Retired command modules and aliases absent | Runtime/import/path absence gates and installed-entry-point test |
| 2 | Canonical configuration ownership | Deleted legacy loader/read shapes; shipped configs and schema validate canonically |
| 3 | One installed command family | Only `trend` and intentional `trend-llm-proxy`; retired executable-token gate |
| 4 | Canonical orchestration and data concerns | Shared entrypoint preparation, shared loader skeleton, direct stage/runner ownership |
| 5 | Orphan launchers/apps/examples removed | Duplicate launcher and retired examples/apps absent; current structural index gate |
| 6 | Test-only production surfaces removed | `io.validators` deleted; `CashPolicy` has one import owner; report/export fallback removed |
| 7 | Monkeypatch compatibility seams removed | Pipeline and multi-period forwarding/synchronization symbols absence-gated |
| 8 | Final supported-surface gate | Full validation record below |

## Repository-wide legacy-token classification

The final scan covers source, tests, scripts, examples, active docs, workflows/actions, notebooks outside `old/`, and root runtime text. Remaining legacy-like text is limited to: explicit negative assertions in the absence gates; dependency/package and container image names such as `trend-model`; algorithmic terms such as dependency aliases and time-frequency compatibility; and archived historical evidence under excluded archive roots. There are no unexplained product/runtime imports, launchers, aliases, compatibility wrappers, or test-only production modules.

## Validation record

Validation completed on the final working tree before commit:

| Gate | Command | Result |
| --- | --- | --- |
| Complete demo and supported runtime path | `.venv/bin/python scripts/run_multi_demo.py` | Passed. Generated the demo inputs, completed four multi-period results, exercised report/export behavior and repeated CLI runs, ran `trend check` directly and through `TREND_CFG`, ran the quick check, verified the dependency lock, and completed the official full test runner. |
| Full repository test suite | `./scripts/run_tests.sh` (also invoked by the complete demo) | 5,864 passed, 6 skipped; 87.37% coverage against an 85% gate. The suite includes the unified legacy-absence, isolated-wheel/installed-CLI, Streamlit, export/report, configuration, Monte Carlo, scheduler, and pipeline regressions. |
| Development gate | `./scripts/dev_check.sh --verbose` | Passed syntax, import, Black, critical Flake8, mypy, and keepalive-harness checks. |
| Full Ruff lint | `.venv/bin/ruff check src tests scripts streamlit_app` | Passed. |
| Black format | `.venv/bin/black --check src tests scripts streamlit_app` | Passed; 978 files unchanged. |
| Source type check | `.venv/bin/mypy src/ --follow-imports=silent --ignore-missing-imports` | Passed; no issues in 198 source files. |
| Installed command contracts | `.venv/bin/python -m trend.cli --help`; `TREND_CFG=config/demo.yml .venv/bin/python -m trend.cli check`; `.venv/bin/python -m trend.cli mc --help` | Passed. |
| Patch integrity | `git diff --check` | Passed. |

Phase 8 is closed by these gates and by the structural regression tests that fail if a retired path, import, façade, schema read, command, example, or page alias returns.

## Remaining work

None. Any future reintroduction of a retired path, symbol, command, schema read, example, page alias, or test-only runtime seam is expected to fail the unified legacy-surface gate.
