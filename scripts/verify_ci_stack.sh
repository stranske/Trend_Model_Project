#!/usr/bin/env bash
set -euo pipefail

# Verify local equivalents of CI checks to triage failures before pushing
# - Black (format), Ruff (lint), MyPy (type check), Pytest (unit tests)
# - Optional: Docker build and smoke test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d .venv ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate || true
fi

echo "==> Running Black (check)"
black --check . || { echo "Black failed"; exit 1; }

echo "==> Running Ruff"
ruff check . || { echo "Ruff failed"; exit 1; }

echo "==> Running MyPy (strict where configured)"
mypy || { echo "MyPy failed"; exit 1; }

echo "==> Running unit tests"
pytest -m "not quarantine and not slow" --maxfail=1 -q || { echo "Tests failed"; exit 1; }

if [ "${1:-}" = "--docker" ]; then
  echo "==> Building Docker image"
  docker build -t trend-model:ci .
  echo "==> Running tests in Docker image"
  docker run --rm trend-model:ci pytest -q
  echo "==> Smoke testing health endpoint"
  cid=$(docker run -d -p 8000:8000 trend-model:ci)
  attempt=1; max_attempts=10
  until curl -fs http://localhost:8000/health | grep -q OK; do
    echo "Health check attempt $attempt failed"; attempt=$((attempt+1)); [ $attempt -le $max_attempts ] || { echo "Health check failed"; docker logs "$cid" || true; docker rm -f "$cid"; exit 1; }; sleep 1;
  done
  docker rm -f "$cid"
fi

echo "All local checks passed."
