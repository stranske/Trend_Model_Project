#!/bin/bash
# Run tests after ensuring dependencies are installed.
# Sets PYTHONHASHSEED for reproducible test results.

set -euo pipefail

# Set hash seed before Python starts for reproducible results
export PYTHONHASHSEED=0

pip install -r requirements.txt pytest
PYTHONPATH="./src" pytest --cov trend_analysis --cov-branch "$@"
