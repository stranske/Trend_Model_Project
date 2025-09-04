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
		pip install -e ".[dev]"
		
		# Try to install the package in editable mode for CLI access
		pip install -e . || echo "Warning: Package installation failed, CLI available via scripts/trend-model"
		
		# Install pre-commit hooks
		if ! pre-commit install --install-hooks; then
			echo "::warning::pre-commit install --install-hooks failed, but continuing. Git hooks may not be available."
		fi
		
		# Ensure CLI wrapper script is executable
		if ! chmod +x scripts/trend-model; then
			echo "::warning::chmod +x scripts/trend-model failed, but continuing. CLI wrapper may not be executable."
		fi
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
pip install -e ".[dev]"

# Try to install the package in editable mode for CLI access
pip install -e . || echo "Warning: Package installation failed, CLI available via scripts/trend-model"

# Install pre-commit hooks so formatting runs locally before commits
if ! pre-commit install --install-hooks; then
	echo "::warning::pre-commit install --install-hooks failed, but continuing. Git hooks may not be available."
fi

# Ensure CLI wrapper script is executable
if ! chmod +x scripts/trend-model; then
	echo "::warning::chmod +x scripts/trend-model failed, but continuing. CLI wrapper may not be executable."
fi

echo "Environment setup complete. Activate later with 'source $ENV_DIR/bin/activate'."
echo "CLI available as: 'trend-model' (if installed) or './scripts/trend-model'"
