#!/bin/bash
# Run tests after ensuring dependencies are installed.
# Sets PYTHONHASHSEED for reproducible test results.

set -euo pipefail

# Set hash seed before Python starts for reproducible results
export PYTHONHASHSEED=0

pip install -r requirements.txt pytest coverage

# Select coverage profile (defaults to "core" if not provided)
PROFILE="${COVERAGE_PROFILE:-core}"

# Run pytest under coverage and capture exit code so we can handle the "no tests" case
set +e
PYTHONPATH="./src" coverage run --rcfile ".coveragerc.${PROFILE}" -m pytest --maxfail=1 --disable-warnings "$@"
status=$?
set -e

if [[ "$status" -eq 5 ]]; then
  echo "No tests were collected or ran. This may be due to test filters or missing/misnamed tests."
  exit 1
fi

coverage report -m

exit "$status"
