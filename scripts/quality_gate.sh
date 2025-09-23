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
