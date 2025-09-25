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

# Type checking (mypy) – fail fast if types regress (core + app)
if command -v mypy >/dev/null 2>&1; then
  echo "Running mypy (core + app)..." >&2
  if ! mypy --config-file pyproject.toml src/trend_analysis src/trend_portfolio_app >/dev/null; then
    echo "mypy failed. To reproduce locally run: mypy --config-file pyproject.toml src/trend_analysis src/trend_portfolio_app" >&2
    exit 1
  fi
else
  echo "mypy not installed; skipping type check (install with pip install -e .[dev])" >&2
fi

# If workflows changed, run local workflow lint (actionlint) if script exists
if git diff --name-only HEAD~1 2>/dev/null | grep -q '^.github/workflows/'; then
  if [[ -x scripts/workflow_lint.sh ]]; then
    echo "Detected workflow changes; running workflow lint..." >&2
    if ! scripts/workflow_lint.sh; then
      echo "Workflow lint failed" >&2
      exit 1
    fi
  else
    echo "Workflow changes detected but scripts/workflow_lint.sh missing; skipping" >&2
  fi
fi

# If Dockerfile or requirements.lock changed, optionally run docker smoke
if git diff --name-only HEAD~1 2>/dev/null | grep -Eq '^(Dockerfile|requirements.lock)'; then
  if [[ -x scripts/docker_smoke.sh ]]; then
    echo "Detected Docker-related changes; running docker smoke test..." >&2
    if ! scripts/docker_smoke.sh; then
      echo "Docker smoke test failed" >&2
      exit 1
    fi
  else
    echo "Docker changes detected but scripts/docker_smoke.sh missing; skipping" >&2
  fi
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
