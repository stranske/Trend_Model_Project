#!/usr/bin/env bash
set -euo pipefail

# Archived helper. Prefer scripts/run_streamlit.sh.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

exec streamlit run "$ROOT/src/trend_portfolio_app/app.py" --server.headless true "$@"
