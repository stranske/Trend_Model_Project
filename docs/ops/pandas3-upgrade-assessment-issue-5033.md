# Pandas 3 Upgrade Assessment (Issue #5033)

## Summary
This assessment evaluated Dependabot PR [#5011](https://github.com/stranske/Trend_Model_Project/pull/5011), which bumps `pandas` from `2.3.3` to `3.0.1`.

Decision: **defer merge of #5011** until identified compatibility blockers are resolved.

## Evidence
### 1) Dependency solver conflict in Gate
- Workflow: `Gate`
- Run: [22312550893](https://github.com/stranske/Trend_Model_Project/actions/runs/22312550893)
- Failed job: `LLM Dependency Compatibility`
- Log excerpt shows:
  - `trend-model ... depends on pandas==3.0.1`
  - `streamlit 1.54.0 depends on pandas<3 and >=1.4.0`
  - `ERROR: ResolutionImpossible`

Impact: the current dependency set cannot be installed consistently with pandas 3.

### 2) Runtime behavior break from pandas offset alias change
- Workflow: `Gate`
- Run: [22312550893](https://github.com/stranske/Trend_Model_Project/actions/runs/22312550893)
- Failed job: `Config Coverage Check`
- Failure path:
  - `src/data/contracts.py` (`_check_frequency`)
  - `src/trend_analysis/data.py`
  - `scripts/check_config_coverage.py`
- Error in logs: `'M' is no longer supported for offsets. Please use 'ME' instead.`

Impact: current frequency handling relies on alias behavior changed in pandas 3, causing validation failures.

## Conclusion
PR #5011 should remain unmerged until the following are completed:
1. Resolve `streamlit` compatibility with pandas 3 (either by upgrading streamlit to a compatible release or by revising dependency strategy).
2. Migrate frequency alias handling (`M` -> `ME`) with backward-compatible validation where needed.
3. Re-run Gate/MC-Viz checks and confirm green status on the pandas-3 branch.

## Follow-up
- Tracking issue: [#5033](https://github.com/stranske/Trend_Model_Project/issues/5033)
- This document is the formal assessment artifact for that issue.
