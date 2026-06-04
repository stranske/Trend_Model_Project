#!/usr/bin/env bash
# Parse-check tracked shell scripts under scripts/.

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

failed=0
count=0
while IFS= read -r script; do
  count=$((count + 1))
  if ! bash -n "$script"; then
    echo "Shell parse check failed: $script" >&2
    failed=1
  fi
done < <(git ls-files 'scripts/*.sh' | sort)

if [[ "$count" -eq 0 ]]; then
  echo "No tracked scripts/*.sh files found."
  exit 0
fi

if [[ "$failed" -ne 0 ]]; then
  exit "$failed"
fi

echo "Shell parse check passed for $count tracked scripts/*.sh files."
