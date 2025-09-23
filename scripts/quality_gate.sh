#!/usr/bin/env bash
# Combined local quality gate: style + validation (fast) + optional full checks.
# Usage:
#   ./scripts/quality_gate.sh          # style + fast validation
#   ./scripts/quality_gate.sh --full   # also run comprehensive branch checks
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FULL=0
if [[ "${1:-}" == "--full" ]]; then
  FULL=1
fi

# Style Gate mirror
./scripts/style_gate_local.sh

# Type checking (mypy) – fail fast if types regress
if command -v mypy >/dev/null 2>&1; then
  echo "Running mypy (strict) on core package..." >&2
  if ! python -m mypy --config-file pyproject.toml src/trend_analysis >/dev/null; then
    echo "mypy failed. Fix type errors or run: python -m mypy --config-file pyproject.toml src/trend_analysis" >&2
    exit 1
  fi
else
  echo "mypy not installed; skipping type check (install with pip install -e .[dev])" >&2
fi

# Fast validation (adaptive)
if [[ -x ./scripts/validate_fast.sh ]]; then
  echo "Running adaptive validation..." >&2
  ./scripts/validate_fast.sh --fix
fi

if [[ $FULL -eq 1 ]]; then
  if [[ -x ./scripts/check_branch.sh ]]; then
    echo "Running comprehensive branch check..." >&2
    ./scripts/check_branch.sh --fast --fix
  fi
fi

echo "Quality gate completed successfully." >&2
