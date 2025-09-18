#!/bin/bash
# Run tests after ensuring dependencies are installed.
# Sets PYTHONHASHSEED for reproducible test results.

set -euo pipefail

# Set hash seed before Python starts for reproducible results
export PYTHONHASHSEED=0

pip install -r requirements.txt pytest coverage

# Select coverage profile (defaults to "core" if not provided)
PROFILE="${COVERAGE_PROFILE:-core}"

# Validate coverage profile file exists
if [[ ! -f ".coveragerc.${PROFILE}" ]]; then
  echo "Invalid coverage profile: ${PROFILE}. File .coveragerc.${PROFILE} not found."
  exit 1
fi
# Run pytest under coverage and capture exit code so we can handle the "no tests" case
set +e
PYTHONPATH="./src" coverage run --branch --rcfile ".coveragerc.${PROFILE}" -m pytest --disable-warnings "$@"
status=$?
set -e

if [[ "$status" -eq 5 ]]; then
  echo "No tests were collected or ran. This may be due to test filters or missing/misnamed tests."
  exit 1
fi

if [[ "$status" -eq 1 ]]; then
  echo "Detected test failures. Retrying failed tests once..."
  set +e
  PYTHONPATH="./src" coverage run --branch --rcfile ".coveragerc.${PROFILE}" --append -m pytest --disable-warnings --failed-first --maxfail=1 "$@"
  retry_status=$?
  set -e

  if [[ "$retry_status" -eq 0 ]]; then
    echo "Retry succeeded; flaky failures resolved on re-run."
    status=0
  else
    echo "Retry failed with status ${retry_status}. Persistent failures detected."
    status=$retry_status
  fi
fi

coverage report -m

exit "$status"
