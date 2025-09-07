#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
for f in .github/workflows/*.y*ml; do
  # strip non-ASCII and normalize line endings
  perl -CSDA -pe 's/[^\x09\x0A\x0D\x20-\x7E]//g' "$f" > "$f.clean"
  mv "$f.clean" "$f"
done
