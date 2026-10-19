# Monte Carlo component-to-production integration audit

Source issue #6100. Audited base: `917d3dea8b52b1d804fc69f7fe003924147c2a81`.
This inventory covers only sampled costs to net outputs, score reuse during real
strategy evaluation, and canonical CLI cost preflight. It introduces no production
repair, algorithm change, automatic finding upload, or general coverage campaign.

| Boundary and contract | Producer to public consumer | Evidence and disposition |
| --- | --- | --- |
| Sampled costs debit actual portfolio returns once and reach metrics and NAV. | `CostProcess.sample` → `_maybe_sample_costs` → `_evaluate_strategy` net returns/statistics → evaluation NAV and `metadata.nav_paths`. | **Confirmed, executed**: the joined real-run test below checks positive debits, exact gross-minus-debit returns, CAGR, evaluation NAV and exported canonical NAV. Existing `test_costs.py::test_cost_debits_reconcile_to_turnover` also checks the debit arithmetic. Original repair: [#6092 / PR #6113](https://github.com/stranske/Trend_Model_Project/pull/6113), merged `0b87f8e9b5e40ff929322c2905a86ec6f372da20`. |
| Compatible strategies reuse metrics on one path, while still executing actual strategy evaluations. | `_generate_path_context` shared metric cache → `_evaluate_strategy` cache context → real `run_simulation` and rank-selection metric kernel. | **Confirmed, executed**: one versus three real strategies, with sampled costs enabled, requires positive and equal wrapped kernel-call counts, three evaluations and reconciled public outputs. The wrapper calls the actual kernel. Original repair: [#6094 / PR #6116](https://github.com/stranske/Trend_Model_Project/pull/6116), merged `967167a53ae67eaefb82b4dc5863d2b521c1f55f`. |
| CLI preflight accepts canonical documented costs and rejects invalid distribution parameters before output creation. | Literal `docs/phase-3/MonteCarlo.md` cost fragment → scenario loader → `validate_mc_scenario` → `CostProcess.from_config` → `mc validate` exit/output. | **Confirmed, executed**: the exact documented fragment succeeds; changing only calm sigma to -1 returns exit 1 with the cost/sigma error. Neither case creates another file. No parser, CLI or validation result is mocked. Original repair: [#6095 / PR #6114](https://github.com/stranske/Trend_Model_Project/pull/6114), merged `fb3277a5e22a736895d5775547dc8e65770639be`. |
| Standalone path-cache examples do not alone establish production integration. | `evaluate_strategies_for_path` and `PathContextCache` direct tests. | **Intentional helper-only controls**: retain their scope and performance checks; do not cite them as proof that the runner consumes the helper. Production proof above exercises the runner's actual metric-cache path. |
| NAV bundle cardinality differs from strategy-evaluation cardinality. | Multiple strategy evaluations → canonical per-path NAV bundle. | **Intentional contract**: the joined test checks one exported path column and reconciles it to each identical-strategy evaluation. The first local draft incorrectly assumed one column per strategy; its failing index access was a test error, not a product defect. |

## Executable prevention and limits

```sh
PYTHONPATH=src python -m pytest tests/monte_carlo/test_runner.py tests/test_cli_mc.py -q
```

The new cases are
`test_audit_shared_scores_and_sampled_costs_reach_real_strategy_outputs` and the
two `test_audit_documented_cost_fragment_reaches_cli_preflight` cases. They use
deterministic one-path inputs, real engines and actual CLI validation. Existing
cache compatibility/fold/release tests and cost multi-period/custom-weight tests
retain their separate responsibilities; this bounded audit does not claim every
parallel simulation, stochastic seed, date alignment or renderer is certified.

Negative controls disconnect three actual production consumers independently:
remove the net debit, replace the shared strategy metric cache with a new empty
cache, and skip the CLI's actual canonical cost parser. Each must fail its named
new regression. Restore every source file byte-identically before the full gate.
The committed [evidence directory](evidence/mc-integration-6100/) retains the actual
transcripts and restoration hashes. Each independent control failed one new test;
the restored full named gate passed **124 tests**. Paths use `$CHECKOUT` and
`$REPO_VENV` for portable provenance.

No additional confirmed gap was found within these three seams. No source issue
is closed by this report, and the linked merges are lineage rather than a new
post-merge acceptance verdict. Hosted CI, review and original-source verification
remain separate.
