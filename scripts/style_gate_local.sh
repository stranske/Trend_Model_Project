#!/usr/bin/env bash
# Mirror of CI Style Gate (Black + Ruff) for local verification.
# Ensures developers run the exact pinned versions before pushing.
# Usage: ./scripts/style_gate_local.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

if [[ ! -f .github/workflows/autofix-versions.env ]]; then
  echo "Missing .github/workflows/autofix-versions.env; aborting." >&2
  exit 1
fi

# Export versions (RUFF_VERSION, BLACK_VERSION, etc.)
set -a
source .github/workflows/autofix-versions.env
set +a

echo "Installing pinned formatter versions: black==${BLACK_VERSION} ruff==${RUFF_VERSION}" >&2
python -m pip install -q "black==${BLACK_VERSION}" "ruff==${RUFF_VERSION}" >/dev/null

echo "Running Black check..." >&2
if ! black --check .; then
  echo "Black check failed. Run: black ." >&2
  exit 1
fi

echo "Running Ruff check (no fixes)..." >&2
if ! ruff check .; then
  echo "Ruff check failed. Run: ruff check --fix ." >&2
  exit 1
fi

echo "Style Gate local mirror: PASS" >&2
