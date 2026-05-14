# Raise test coverage to 89%

**Status:** Pending

CI coverage threshold has been temporarily reduced to keep the build green while additional tests are developed.

Once overall test coverage meets or exceeds **89%**, restore the original coverage gate and remove the temporary reduction.

## Steps
- Generate branch coverage report with `pytest --cov=src --cov-report=html --cov-branch`.
- Identify uncovered code paths and add tests targeting them.
- Increase `--cov-fail-under` in CI back to 89% when coverage improves.

## Reference
- Current threshold: 80% (temporary). Set by `coverage-min: "80"` in `.github/workflows/pr-00-gate.yml` and `coverage-min: '80'` in `.github/workflows/ci.yml`.
- Target threshold: 89%
