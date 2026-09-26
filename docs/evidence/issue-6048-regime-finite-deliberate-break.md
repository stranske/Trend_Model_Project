# Deliberate-break evidence: regime finite-value guards (#6048 / #6025)

Linked issue: `stranske/Trend_Model_Project#6048` (verification follow-up for merged PR #6044 / issue #6025).

## Mutation (temporary, reverted before merge)

In `src/trend_analysis/regimes.py`, the `np.isfinite` check inside `_require_finite_float` was
temporarily disabled (`if False and not np.isfinite(num)`) so `normalise_settings` would accept
non-finite `neutral_band` values.

Production code on this branch still rejects non-finite regime controls; this file records the
RED/GREEN transcript only.

## RED — disable finite guard, then `pytest tests/test_regime_annualise.py::test_regime_control_values_reject_non_finite_and_string_boolean -q --no-cov`

```
F                                                                        [100%]
=================================== FAILURES ===================================
_______ test_regime_control_values_reject_non_finite_and_string_boolean ________

    def test_regime_control_values_reject_non_finite_and_string_boolean() -> None:
        with pytest.raises(ValueError, match="must be a boolean"):
            normalise_settings({"enabled": "false", "cache": "false"})
>       with pytest.raises(ValueError, match="must be a finite number"):
E       Failed: DID NOT RAISE ValueError

tests/test_regime_annualise.py:101: Failed
=========================== short test summary info ============================
FAILED tests/test_regime_annualise.py::test_regime_control_values_reject_non_finite_and_string_boolean
1 failed in 8.25s
```

The boolean-flag half of the test still passes; only the `neutral_band: nan` finite guard fails to
fire when `_require_finite_float` is bypassed.

## GREEN — restore guard, same command

```
.                                                                        [100%]
1 passed in 4.58s
```

## Verification command (this PR)

`pytest tests/test_regime_annualise.py::test_regime_control_values_reject_non_finite_and_string_boolean -q --no-cov`

Recorded at branch tip `19de99c7` (post-#6047 evidence merge on `main`).
