# Settings snapshot reconciliation: source issue 6099

Observed source `95fc7f1c`; no historical generator run or report regeneration.
The original 2025-12-21 report payloads and SUMMARY counts are SHA-256 pinned
in [provenance.json](../settings_evidence/provenance.json), with their generation
source revision explicitly UNKNOWN. All seven historical FAILs remain FAIL.
Seven README rows separate entry point, fixture/config, mode and metric. The
June 2026 audit is treated as historical, including its stale weighting result.

Current focused observations (not equivalent historical experiments):

- Four directional cases PASS: `max_weight_down`, `rank_n_down`, `cost_up`,
  `weighting_risk_parity`. They use committed `config/demo.yml` and
  `demo/demo_returns.csv`, direct config patches and the real computational
  harness. The cost assertion permits equality; it is not proof of increasing
  old UI-path total_costs. Weighting clears custom_weights explicitly.
- Four boundary cases PASS: the two direct API signal tests (including
  20/60-window signal frames), top_pct rank-n non-cap behavior, and the pure
  synthetic over-leveraged-book invariant. The latter does not run an engine.
- The provenance/timestamp module passes. Removing the actual README historical
  snapshot marker makes its new regression fail; exact-byte restoration returns
  GREEN. This detects publication of unsupported current-evidence language.

The JSON receipt retains the source SHA and demo input/config hashes. Local
commands use the repository-selected `.venv/bin/python` (Anaconda CPython3.12.2)
and the existing read-only `Workflows/packages/app-baseline-kit` on PYTHONPATH.
Initial collection without that dependency failed; it was a setup failure, not a
regression result. Existing Pandas4 deprecations and missing historical Trend
Universe metadata warnings were observed. The actual current baseline fixture
was the committed demo CSV, not the old Trend Universe file.

```sh
PYTHONPATH=/path/to/Workflows/packages/app-baseline-kit:src python -m pytest -o addopts= tests/baseline/test_directional.py -k 'max_weight_down or rank_n_down or cost_up or weighting_risk_parity' -q
PYTHONPATH=/path/to/Workflows/packages/app-baseline-kit:src python -m pytest -o addopts= tests/test_api_signal_settings.py tests/baseline/test_rank_fund_count_limits.py::test_top_pct_rank_n_does_not_cap_percentage_selection tests/baseline/test_invariants.py::test_over_leveraged_book_fails_only_the_leverage_invariant -q
python -m pytest -o addopts= tests/scripts/test_generate_settings_evidence_timestamps.py -q
```

`-o addopts=` isolates functional evidence; native Gate coverage remains separate.
No UI wiring, complete historical replay, full-suite or hosted deployment PASS
is inferred. Uncovered comparable-experiment gaps remain explicit in the README;
no new implementation defect or follow-up issue is inferred or filed.
