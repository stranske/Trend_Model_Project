#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="$ROOT_DIR/streamlit_app/app.py"

# Validation function for security and error handling
validate_streamlit_setup() {
    local app_path="$1"
    
    # Validate APP_PATH exists and is readable
    if [[ ! -f "$app_path" ]]; then
        echo "ERROR: Streamlit app not found at: $app_path" >&2
        echo "Expected location: $ROOT_DIR/streamlit_app/app.py" >&2
        return 1
    fi
    
    if [[ ! -r "$app_path" ]]; then
        echo "ERROR: Cannot read Streamlit app at: $app_path" >&2
        echo "Check file permissions." >&2
        return 1
    fi
    
    # Sanitize and validate APP_PATH to prevent path traversal
    local resolved_path
    resolved_path="$(realpath "$app_path" 2>/dev/null)"
    if [[ -z "$resolved_path" ]]; then
        echo "ERROR: Failed to resolve APP_PATH with realpath." >&2
        return 1
    fi
    local expected_path
    expected_path="$(realpath "$ROOT_DIR/streamlit_app/app.py" 2>/dev/null || echo "$ROOT_DIR/streamlit_app/app.py")"
    if [[ "$resolved_path" != "$expected_path" ]]; then
        echo "ERROR: APP_PATH security validation failed." >&2
        echo "Expected: $expected_path" >&2
        echo "Resolved: $resolved_path" >&2
        return 1
    fi
    
    # Validate streamlit command is available
    if ! command -v streamlit >/dev/null 2>&1; then
        echo "ERROR: streamlit command not found in PATH." >&2
        echo "Ensure streamlit is installed and available." >&2
        echo "Try: pip install streamlit" >&2
        return 1
    fi
    
    return 0
}

# Ensure venv exists and is activated via the repo script
if [[ ! -d "$ROOT_DIR/.venv" ]]; then
	echo "No .venv found. Bootstrapping via scripts/setup_env.sh..."
	if ! bash "$ROOT_DIR/scripts/setup_env.sh"; then
		echo "ERROR: Failed to bootstrap environment." >&2
		exit 1
	fi
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

# Validate setup before executing streamlit
if ! validate_streamlit_setup "$APP_PATH"; then
    echo "ERROR: Validation failed. Cannot start Streamlit app." >&2
    exit 1
fi

# Set PYTHONPATH to include project root and src for module resolution
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Execute streamlit with proper error handling
echo "Starting Streamlit app: $APP_PATH"
exec streamlit run "$APP_PATH" "$@"
