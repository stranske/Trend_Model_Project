#!/usr/bin/env bash
# Local workflow lint using actionlint (mirrors maint-36-actionlint.yml)
# Usage: ./scripts/workflow_lint.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR=".cache/actionlint"
VERSION="1.6.27"
BIN="$CACHE_DIR/actionlint"

mkdir -p "$CACHE_DIR"
if [[ ! -x "$BIN" ]]; then
  echo "Downloading actionlint $VERSION" >&2
  curl -sSfL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s "$VERSION"
  mv actionlint "$BIN"
fi

echo "Running actionlint ($VERSION)" >&2
"$BIN" -color || {
  echo "actionlint failed" >&2
  exit 1
}

echo "Workflow lint: PASS" >&2
