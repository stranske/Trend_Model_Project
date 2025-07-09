#!/bin/bash
# Run tests after ensuring dependencies are installed.

set -euo pipefail

pip install -r requirements.txt pytest
PYTHONPATH="./src" pytest --cov trend_analysis --cov-branch "$@"
