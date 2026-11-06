#!/bin/bash
# Run tests after ensuring dependencies are installed.
# Sets PYTHONHASHSEED for reproducible test results.

set -euo pipefail

# Ensure we are operating inside a writable virtual environment so dependency
# installs do not require elevated permissions.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  VENV_DIR="${VENV_DIR:-.venv}"
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

# Set hash seed before Python starts for reproducible results
export PYTHONHASHSEED=0

python -m pip install --upgrade pip
if [[ -z "${TREND_MODEL_SITE_CUSTOMIZE:-}" ]]; then
  export TREND_MODEL_SITE_CUSTOMIZE=1
fi
# Historically this used `uv pip sync requirements.lock` to enforce lockfile-pinned
# dependencies; use pip here so test execution does not require a uv executable.
python -m pip install -r requirements.lock
# requirements.lock intentionally omits app-baseline-kit because its source is
# an unversioned Workflows git URL. Resolve the declared dev extra after the
# locked packages so a clean test environment includes that required harness.
python -m pip install -e ".[dev]"

# Select coverage profile (defaults to "core" if not provided)
PROFILE="${COVERAGE_PROFILE:-core}"

# Validate coverage profile file exists
if [[ ! -f ".coveragerc.${PROFILE}" ]]; then
  echo "Invalid coverage profile: ${PROFILE}. File .coveragerc.${PROFILE} not found."
  exit 1
fi

# Build pytest arguments
if [[ -n "${TREND_SKIP_COVERAGE_REPORT:-}" ]]; then
  # Narrow selections can satisfy pytest but not repository coverage thresholds; use for focused gates only.
  PYTEST_ARGS=("--maxfail=1" "--disable-warnings")
else
  PYTEST_ARGS=("--maxfail=1" "--disable-warnings" "--cov=src" "--cov-branch" "--cov-config=.coveragerc.${PROFILE}")
fi

# Enable parallel execution if pytest-xdist is available
if [[ -z "${TREND_SKIP_COVERAGE_REPORT:-}" ]] && python -c "import xdist" 2>/dev/null; then
  PYTEST_ARGS+=("-n" "auto")
  echo "Running tests in parallel with pytest-xdist..."
elif [[ -n "${TREND_SKIP_COVERAGE_REPORT:-}" ]]; then
  echo "Running tests sequentially (TREND_SKIP_COVERAGE_REPORT excludes parallel default)"
else
  echo "Running tests sequentially (install pytest-xdist for parallel execution)"
fi

# Run pytest with coverage and capture exit code so we can handle the "no tests" case
set +e
PYTHONPATH="./src" pytest "${PYTEST_ARGS[@]}" "$@"
status=$?
set -e

if [[ "$status" -eq 5 ]]; then
  echo "No tests were collected or ran. This may be due to test filters or missing/misnamed tests."
  exit 1
fi

if [[ -z "${TREND_SKIP_COVERAGE_REPORT:-}" ]]; then
  # Generate coverage report
  coverage report -m
fi

exit "$status"
