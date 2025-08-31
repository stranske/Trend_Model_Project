#!/bin/bash
# Setup a local Python virtual environment and install dependencies.

# Detect if this script is being sourced (so we don't leak set -euo into the parent shell)
# shellcheck disable=SC2039
if (return 0 2>/dev/null); then
	# Sourced: run the install in a child shell to keep strict modes local,
	# then activate the venv in the current shell.
	(
		set -euo pipefail
		ENV_DIR=".venv"
		python3 -m venv "$ENV_DIR"
		# shellcheck source=/dev/null
		source "$ENV_DIR/bin/activate"
		pip install --upgrade pip
		pip install -r requirements.txt
		pip install pre-commit black ruff mypy
		
		# Try to install the package in editable mode for CLI access
		pip install -e . || echo "Warning: Package installation failed, CLI available via scripts/trend-model"
		
		# Install pre-commit hooks
		pre-commit install --install-hooks || true
		
		# Ensure CLI wrapper script is executable
		chmod +x scripts/trend-model || true
	)
	# Now activate in the current shell so the user can keep working
	# shellcheck disable=SC1091
	. ".venv/bin/activate"
	echo "Environment ready and activated."
	echo "CLI available as: 'trend-model' (if installed) or './scripts/trend-model'"
	return 0 2>/dev/null || exit 0
fi

# Executed normally (recommended): strict mode is safe here
set -euo pipefail

ENV_DIR=".venv"

python3 -m venv "$ENV_DIR"
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip install pre-commit black ruff mypy

# Try to install the package in editable mode for CLI access
pip install -e . || echo "Warning: Package installation failed, CLI available via scripts/trend-model"

# Install pre-commit hooks so formatting runs locally before commits
pre-commit install --install-hooks || true

# Ensure CLI wrapper script is executable
chmod +x scripts/trend-model || true

echo "Environment setup complete. Activate later with 'source $ENV_DIR/bin/activate'."
echo "CLI available as: 'trend-model' (if installed) or './scripts/trend-model'"
