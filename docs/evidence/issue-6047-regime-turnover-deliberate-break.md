# Deliberate-break evidence: regime turnover parsing (#6047 / #6024)

Linked issue: `stranske/Trend_Model_Project#6047` (verification follow-up for merged PR #6043 / issue #6024).

## Mutation (temporary, reverted before merge)

In `src/trend_analysis/pipeline_runner.py`, the fail-closed call to
`parse_regime_turnover_caps` was temporarily wrapped so `CoreConfigError` was swallowed
and `parsed_turnover` became `None`, restoring the pre-#6043 fallback behavior.

Production code on this branch still fails closed; this file records the RED/GREEN
transcript only.

## RED — restore fallback, then `pytest tests/test_pipeline_entrypoints.py::test_run_from_config_rejects_invalid_regime_turnover_cap -q --no-cov`

```
FAILED tests/test_pipeline_entrypoints.py::test_run_from_config_rejects_invalid_regime_turnover_cap[turnover0]
FAILED tests/test_pipeline_entrypoints.py::test_run_from_config_rejects_invalid_regime_turnover_cap[bad]
FAILED tests/test_pipeline_entrypoints.py::test_run_from_config_rejects_invalid_regime_turnover_cap[oops]
3 failed in 25.65s
```

Representative failure (`turnover0` = `{"mystery": 0.1}`):

```
AssertionError: selection should not run when turnover parsing fails
```

The test expects `CoreConfigError` before universe selection; the temporary fallback let
selection run instead of failing closed.

## GREEN — restore fail-closed parsing, same command

```
...                                                                      [100%]
3 passed in 17.65s
```

## Verification command (this PR)

`pytest tests/test_pipeline_entrypoints.py::test_run_from_config_rejects_invalid_regime_turnover_cap -q --no-cov`

Recorded at branch tip `cbbadb7d` (post-#6046 merge on `main`).
