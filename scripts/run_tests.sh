#!/bin/bash
# Run tests after ensuring dependencies are installed.
# Sets PYTHONHASHSEED for reproducible test results.

set -euo pipefail

# Set hash seed before Python starts for reproducible results
export PYTHONHASHSEED=0

pip install -r requirements.txt pytest

# Run pytest and capture exit code so we can handle the "no tests" case
set +e
PYTHONPATH="./src" pytest --maxfail=1 --disable-warnings --cov trend_analysis --cov-branch "$@"
status=$?
set -e
if [ "$status" -eq 5 ]; then
  echo "No tests were collected or ran. This may be due to test filters or missing/misnamed tests."
  exit 1
fi

exit "$status"
