# Deliberate-break evidence: `lambda_tc` finite guard (#6046 / #6023)

Linked issue: `stranske/Trend_Model_Project#6046` (verification follow-up for merged PR #6042 / issue #6023).

## Mutation (temporary, reverted before merge)

In `src/trend_analysis/config/models.py`, the `math.isfinite(lam)` guard in the portfolio
`lambda_tc` dict validator was temporarily disabled (`if False and not math.isfinite(lam)`)
so the named test could observe at least one non-finite acceptance failure.

Production code on this branch still includes the guard; this file records the RED/GREEN
transcript only.

## RED — disable finite guard, then `pytest tests/test_config_turnover_validation.py::test_lambda_tc_rejects_non_finite -q --no-cov`

```
F..                                                                      [100%]
=================================== FAILURES ===================================
____________________ test_lambda_tc_rejects_non_finite[nan] ____________________

lam = nan

    @pytest.mark.parametrize("lam", [float("nan"), float("inf"), float("-inf")])
    def test_lambda_tc_rejects_non_finite(lam):
        cfg_dict = make_cfg({"portfolio": {"lambda_tc": lam}})
>       with pytest.raises(Exception):
E       Failed: DID NOT RAISE Exception

tests/test_config_turnover_validation.py:91: Failed
=========================== short test summary info ============================
FAILED tests/test_config_turnover_validation.py::test_lambda_tc_rejects_non_finite[nan]
1 failed, 2 passed in 11.82s
```

(`inf` / `-inf` still fail closed via adjacent range validators; the finite guard is what
`nan` slips through when disabled.)

## GREEN — restore guard, same command

```
...                                                                      [100%]
3 passed in 13.39s
```

## Verification command (this PR)

`pytest tests/test_config_turnover_validation.py::test_lambda_tc_rejects_non_finite -q --no-cov`
