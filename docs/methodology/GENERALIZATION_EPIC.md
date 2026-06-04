# Generalization Epic

Issue #5416 decomposes the manager-of-managers generalization work into bounded children. The current engine is reusable in places, but several surrounding seams still assume the original trend/SPX framing.

## Already Exists

- Cadence aliases for monthly, quarterly, and annual runs already map to pandas end-of-period codes in `src/trend_analysis/multi_period/scheduler.py:17`.
- Dated universe membership and survivorship handling already live in `src/trend_analysis/universe.py`.
- Benchmark labels already map to data columns through `config/defaults.yml:123`.
- The default regime path is implemented by `_compute_regime_series` in `src/trend_analysis/regimes.py`, and remains registered as `binary_threshold` on `BinaryThresholdRegimeModel`.

## Children

### Cadence Generalization

- Current evidence: `src/trend_analysis/multi_period/scheduler.py:17` covers M/Q/A aliases, so the remaining gap is weekly/daily cadence end-to-end rather than re-adding monthly support.
- Proposed seam: extend scheduler frequency validation and downstream period consumers for shorter cadence names.
- Acceptance shape: a weekly or daily fixture runs through period generation and the main analysis path with a named regression test.
- Dependencies: none.

### Convex Constraints

- Current evidence: the weighting interface now has a constrained convex backend from issue #5414.
- Proposed seam: keep additional group/turnover/position-limit constraints behind the weighting registry rather than embedding them in strategy code.
- Acceptance shape: child issues should cite the existing #5414 backend and add one constraint type plus a named test.
- Dependencies: existing issue #5414.

### Pluggable Regime Interface

- Current evidence: `src/trend_analysis/regimes.py` implements the binary return/volatility threshold classifier in `_compute_regime_series`.
- Proposed seam: `RegimeModel` plus `regime_registry`, with the existing classifier registered as `binary_threshold`.
- Acceptance shape: `tests/test_regime_registry.py::test_default_binary_regime_unchanged` proves default output is unchanged, and `tests/test_regime_registry.py::test_registry_dispatches_named_model` proves named dispatch works.
- Dependencies: none.

### Pluggable Cost Interface

- Current evidence: transaction-cost behavior is not exposed as a registry-backed model seam in the weighting/turnover path.
- Proposed seam: a cost-model protocol and registry consumed where turnover costs are computed.
- Acceptance shape: the default cost model preserves current output, while a named toy model changes costs in a focused fixture.
- Dependencies: cadence and weighting callers must pass enough context to the cost model.

### Benchmark And Peer-Index Integration

- Current evidence: `config/defaults.yml:123` maps benchmark labels to columns, but benchmark and peer-index series are not first-class attribution/significance inputs.
- Proposed seam: benchmark/peer series should flow into attribution and reporting through an explicit input contract.
- Acceptance shape: a benchmark fixture affects attribution/significance output with a named test and leaves no-benchmark behavior unchanged.
- Dependencies: issue #5415 factor attribution should be merged before this child expands benchmark attribution.
