#!/bin/bash
# Setup a local Python virtual environment and install dependencies.
set -euo pipefail

ENV_DIR=".venv"

python3 -m venv "$ENV_DIR"
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete. Activate with 'source $ENV_DIR/bin/activate'."
