<!-- bootstrap for codex on issue #3685 -->

## Keepalive Checklist

### Scope
- [x] `analysis/results.py::Results` with: returns, weights, exposures, turnover, costs, and a `metadata` dict (universe, lookbacks, costs, code version).
- [x] Fingerprint: short hash over `Trend Universe Data.csv` and the membership file contents and key columns; store in `metadata`.
- [x] Update summary generation to include metadata and fingerprint.

### Tasks
- [x] Implement `Results` and update call sites.
- [x] Compute and persist input fingerprints.
- [x] Update summary output to display metadata/fingerprint.

### Acceptance criteria
- [x] Results objects include complete fields and a stable input fingerprint.
