#!/usr/bin/env bash
# Mirror of CI style job (Black + Ruff) for local verification.
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

# Run mypy using pinned version if available
if [[ -n "${MYPY_VERSION:-}" ]]; then
  echo "Ensuring pinned mypy==${MYPY_VERSION}" >&2
  TYPES_REQUESTS_SPEC=$(python - <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
dev_requirements = data.get("project", {}).get("optional-dependencies", {}).get(
    "dev", []
)
for entry in dev_requirements:
    if entry.startswith("types-requests=="):
        print(entry)
        break
PY
)
  if [[ -z "${TYPES_REQUESTS_SPEC}" ]]; then
    TYPES_REQUESTS_SPEC="types-requests"
  fi
  python -m pip install -q "mypy==${MYPY_VERSION}" "${TYPES_REQUESTS_SPEC}" pydantic streamlit >/dev/null || {
    echo "Failed to install mypy ${MYPY_VERSION}" >&2
    exit 1
  }
else
  echo "MYPY_VERSION not exported; using existing mypy (if any)" >&2
fi

if command -v mypy >/dev/null 2>&1; then
  echo "Running mypy (core + app)" >&2
  if ! mypy --config-file pyproject.toml src/trend_analysis src/trend_portfolio_app; then
    echo "mypy type check failed." >&2
    exit 1
  fi
else
  echo "mypy not found; skipping type check" >&2
fi
echo "CI style job local mirror: PASS" >&2

