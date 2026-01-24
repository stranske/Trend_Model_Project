#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="$ROOT_DIR/.streamlit"
SECRETS_FILE="$SECRETS_DIR/secrets.toml"

if [[ -f "$SECRETS_FILE" ]]; then
  echo "Secrets already exist at $SECRETS_FILE"
  exit 0
fi

mkdir -p "$SECRETS_DIR"

KEY="${OPENAI_API_KEY:-}"
if [[ -z "$KEY" ]]; then
  read -r -s -p "Enter OPENAI_API_KEY: " KEY
  echo
fi

if [[ -z "$KEY" ]]; then
  echo "No OPENAI_API_KEY provided. Aborting."
  exit 1
fi

cat > "$SECRETS_FILE" <<EOF
# Local secrets (git-ignored). Do not commit real keys.
OPENAI_API_KEY = "$KEY"
EOF

chmod 600 "$SECRETS_FILE"

echo "Wrote $SECRETS_FILE (git-ignored)"
