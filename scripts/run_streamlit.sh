#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="$ROOT_DIR/src/trend_portfolio_app/app.py"

# Ensure venv exists and is activated via the repo script
if [[ ! -d "$ROOT_DIR/.venv" ]]; then
	echo "No .venv found. Bootstrapping via scripts/setup_env.sh..."
	bash "$ROOT_DIR/scripts/setup_env.sh"
else
	# shellcheck source=/dev/null
	source "$ROOT_DIR/.venv/bin/activate"
fi

# Ensure streamlit and streamlit-sortables are available
python - <<'PY' >/dev/null 2>&1 || pip install streamlit
import importlib
importlib.import_module('streamlit')
PY

python - <<'PY' >/dev/null 2>&1 || pip install streamlit-sortables
import importlib
importlib.import_module('streamlit_sortables')
PY

exec streamlit run "$APP_PATH" "$@"
