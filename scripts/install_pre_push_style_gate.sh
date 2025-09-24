#!/usr/bin/env bash
# Install a git pre-push hook that enforces the local Style Gate mirror.
# Usage: ./scripts/install_pre_push_style_gate.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .git ]]; then
  echo "This script must be run at the root of a git repository." >&2
  exit 1
fi

HOOK_DIR=.git/hooks
HOOK_PATH=$HOOK_DIR/pre-push
mkdir -p "$HOOK_DIR"

cat > "$HOOK_PATH" <<'EOF'
#!/usr/bin/env bash
# Auto-generated pre-push hook: run the local Style Gate mirror.
if [[ -x scripts/style_gate_local.sh ]]; then
  scripts/style_gate_local.sh || {
    echo "[pre-push] Style Gate failed; push blocked." >&2
    exit 1
  }
else
  echo "[pre-push] scripts/style_gate_local.sh not found or not executable; skipping style enforcement." >&2
fi
EOF

chmod +x "$HOOK_PATH"

echo "Installed pre-push hook at $HOOK_PATH" >&2
