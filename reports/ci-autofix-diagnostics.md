# CI Autofix Diagnostics

- Timestamp (UTC): 2026-02-11T22:49:31Z
- Mode: autofix from CI failure
- Result: failure not reproducible in local workspace

## Checks run

- `PYTHONPATH=./src pytest tests/workflows -q` -> `205 passed, 1 warning`
- `PYTHONPATH=./src ruff check .` -> `All checks passed!`
- `PYTHONPATH=./src mypy src tests` -> `Success: no issues found in 731 source files`
- `PYTHONPATH=./src pytest -q` -> `5238 passed, 11 skipped`

## Notes

- Checked-in `pytest-junit.xml` reports zero failures/errors.
- No CI failure stack trace or failing assertion was present in the provided local artifacts.
- `black --check .` is significantly slower than other checks in this environment; if the CI failure was a timeout, narrowing Black scope may be required with human review of workflow policy.
