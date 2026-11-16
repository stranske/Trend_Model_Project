## Task reconciliation (PR #6226)

Recent commits on `cursor/issue-6225`:
- `8778ce1f4` — `runner.py`: pass `net_returns if not None else gross_returns` into `_extract_nav_series`
- `ea9f25aa0` — `test_nav_extract_receives_gross_returns_when_scenario_costs_absent`
- `8f4775b89` — `test_nav_user_weighted_named_regression_fails_when_returns_omitted`

## Deliberate break → revert (acceptance criterion)

| Step | Command / state | Result |
|------|-----------------|--------|
| Break | Revert NAV wiring to `returns=net_returns` only (pre-fix) | `test_nav_uses_user_weighted_returns_without_scenario_costs` **FAILED** (distinct NAV / reconciliation assertion) |
| Revert | Restore `nav_returns = net_returns if net_returns is not None else gross_returns` | Same test **PASSED** |
| Guard | `test_nav_user_weighted_named_regression_fails_when_returns_omitted` | **PASSED** (identical equal-weight NAVs + reconciliation failure) |

## Verification (2026-11-16)

```text
pytest tests/monte_carlo/test_runner.py::test_nav_uses_user_weighted_returns_without_scenario_costs -q  → passed
pytest tests/monte_carlo/test_runner.py -q  → 76 passed
```