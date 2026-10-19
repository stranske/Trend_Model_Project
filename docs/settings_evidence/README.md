# Settings Wiring Evidence — historical snapshot

**Snapshot kind: historical. Snapshot date: 2025-12-21. Source revision: UNKNOWN.**

The original PASS/FAIL observations are retained, including all seven FAILs in
[SUMMARY.md](SUMMARY.md). They are neither a latest-run dashboard nor proof of
current UI wiring. SUMMARY generation time/revision are UNKNOWN; its linked
individual reports carry December 21 timestamps. The earlier prefixed reports
and later unprefixed reports are distinct experiments, not interchangeable data.
[provenance.json](provenance.json) records each original report's byte hash,
visible timestamp and source-revision unknown. Banners are the only edits to the
individual historical reports.

## Experiment boundaries

The documented generator path is
[`generate_settings_evidence.py`](../../scripts/generate_settings_evidence.py)
→ [`test_settings_wiring.py`](../../scripts/test_settings_wiring.py)
`run_analysis_with_state` → replicated `_build_config_from_state` →
`api.run_simulation`. It does not drive Streamlit controls. Original executable
revision, complete resolved state, CSV content hash and single/multi-period mode
are UNKNOWN; the dates do not establish those facts. The reported Trend Universe
source is not a frozen reproducible fixture. Current `get_baseline_state` sets
`multi_period_enabled=True` despite its single-period docstring, so that comment
must not be used to infer historical mode.

The [current baseline harness](../../tests/baseline/harness.py) instead patches
engine config directly, starting from [config/demo.yml](../../config/demo.yml).
The [catalog](../../tests/baseline/catalog.yaml) names binding overrides and
metrics; its PASS can establish only that computational experiment. The
[2026-06-11 audit](../reports/multi-period-wiring-audit.md) is another dated
observation with old line references; its report-only weighting result predates
the catalog's custom_weights correction and is not current runtime proof.

## Seven scoped dispositions

Current source inspection and focused execution are recorded at `95fc7f1c` in
[the run receipt](../evidence/issue-6099-settings-provenance.md). No historical
experiment was rerun and no equivalence was established. Every row below retains
its historical FAIL. Config/fixture, entry point, execution mode and metric are
separate comparison axes; unknown historical axes remain UNKNOWN.

| Setting / original values | Historical metric | Current scoped validation / fixture and mode | Disposition |
|---|---|---|---|
| [`leverage_cap`](leverage_cap.md) / 2.0 -> 1.0 | gross_exposure (8066ee3f hash, not a numeric exposure) | [tests/baseline/test_invariants.py::test_over_leveraged_book_fails_only_the_leverage_invariant](../../tests/baseline/test_invariants.py); Synthetic 3x long/short book; pure invariant evaluation, no UI or engine run. | Pure exposure invariant checks a constructed book. No matching end-to-end leverage-slider experiment was run; old UI-path result remains unresolved. Historical FAIL retained; equivalence not established. |
| [`max_weight`](max_weight.md) / 0.20 -> 0.10 | max_position_weight (1.3741 -> 1.3741) | [tests/baseline/test_directional.py::test_directional[max_weight_down]](../../tests/baseline/test_directional.py); config/demo.yml; direct config constraints.max_weight .25 -> .05 with vol_adjust enabled; metric max_weight. | Different cap values and binding fixture; current direct-config direction does not establish old UI translation or historical normalization. Historical FAIL retained; equivalence not established. |
| [`weighting_scheme`](weighting_scheme.md) / equal -> hrp | weight_dispersion (0.3031 -> 0.3031) | [tests/baseline/test_directional.py::test_directional[weighting_risk_parity]](../../tests/baseline/test_directional.py); config/demo.yml; direct portfolio.weighting.name equal -> risk_parity, custom_weights cleared; metric max_weight. | Different weighting names, metric and fixture. The catalog explicitly clears custom_weights, which otherwise masks the engine; no equivalence to old hrp/UI path. Historical FAIL retained; equivalence not established. |
| [`transaction_cost_bps`](transaction_cost_bps.md) / 0 -> 50 | total_costs (0 -> 0) | [tests/baseline/test_directional.py::test_directional[cost_up]](../../tests/baseline/test_directional.py); config/demo.yml; direct portfolio.cost_model.per_trade_bps 0 -> 50; metric ann_return, decrease_or_equal. | Cost key, metric and mode differ. A non-increase in direct-config net return is weaker than proving an increase in UI-path total_costs; old result retained. Historical FAIL retained; equivalence not established. |
| [`trend_window`](trend_window.md) / 63 -> 21 | signal_responsiveness (8066ee3f hash, not a numeric responsiveness) | [tests/test_api_signal_settings.py::test_run_simulation_trend_window_changes_output](../../tests/test_api_signal_settings.py); Synthetic 120-month/three-fund returns; direct api.run_simulation config signals.window 20 -> 60; compares signal frames and spec.window. | Signal-frame sensitivity is observed in a different direct-config single-period experiment; neither a UI-slider claim nor an equivalent hash metric. The earlier signals_trend_window.md is a separate errored 12 -> 6 snapshot. Historical FAIL retained; equivalence not established. |
| [`rank_pct`](rank_pct.md) / .10 -> .30 | num_funds_selected (68 -> 68) | [tests/baseline/test_rank_fund_count_limits.py::test_top_pct_rank_n_does_not_cap_percentage_selection](../../tests/baseline/test_rank_fund_count_limits.py); config/demo.yml; direct top_pct pct=.5/n=5, threshold_hold and max_funds=50; selected_funds count must exceed 5. | Fixed-fraction control proves n does not cap top_pct in that fixture. It does not compare .10/.30 or exercise the old UI-state translation; no equivalent percentage-sensitivity claim. Historical FAIL retained; equivalence not established. |
| [`selection_count`](selection_count.md) / 10 -> 5 | num_funds_selected (68 -> 68) | [tests/baseline/test_directional.py::test_directional[rank_n_down]](../../tests/baseline/test_directional.py); config/demo.yml; direct portfolio.rank.n 10 -> 5; metric num_selected (declared_selected_count when available). | Direct rank-n scenario and historical positive-weight-count extraction differ; count movement is not proof of UI selection_count wiring or of realised holdings in every mode. Historical FAIL retained; equivalence not established. |

## Preservation and future measurements

The timestamp regression remains in
[`tests/scripts/test_generate_settings_evidence_timestamps.py`](../../tests/scripts/test_generate_settings_evidence_timestamps.py).
It now also verifies snapshot markers, original report payload hashes, all seven
scoped dispositions and existing evidence links. Historical timestamps are not
silently made UTC; the existing UTC test governs newly generated timestamps.
No wall-clock expiry is imposed on an intentionally historical snapshot.

`python scripts/generate_settings_evidence.py --list` lists available settings.
The generator writes into this evidence directory and has no output-directory
option. Run new experiments in a separate checkout/artifact location and record
source revision, input hash, full resolved state, mode, command, environment and
metric before comparing results. Do not overwrite these archived FAILs with an
incomparable newer PASS. Gaps in the table require a scoped comparable experiment,
not an inferred implementation defect or an automatically filed follow-up.
