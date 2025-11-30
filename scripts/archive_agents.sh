#!/usr/bin/env bash
# archive_agents.sh — Archive codex instruction files for closed issues
#
# Usage:
#   ./scripts/archive_agents.sh          # Dry-run (shows what would be archived)
#   ./scripts/archive_agents.sh --apply  # Actually move files
#
# Requirements: gh CLI authenticated with repo access

set -euo pipefail

AGENTS_DIR="agents"
ARCHIVE_DIR="archives/agents"
DATE=$(date +%Y-%m-%d)
DRY_RUN=true

if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
fi

cd "$(git rev-parse --show-toplevel)"

echo "=== Agent Archive Script ==="
echo "Date prefix: $DATE"
echo "Mode: $(if $DRY_RUN; then echo 'DRY-RUN (use --apply to move files)'; else echo 'APPLY'; fi)"
echo ""

to_archive=()
to_keep=()

for f in "$AGENTS_DIR"/codex-*.md; do
    [[ -f "$f" ]] || continue
    num=$(basename "$f" | grep -oE '[0-9]+')
    
    # Check issue state
    state=$(gh issue view "$num" --json state -q '.state' 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$state" == "OPEN" ]]; then
        to_keep+=("codex-${num}.md (Issue #${num} OPEN)")
    else
        to_archive+=("$f|$num|$state")
    fi
done

echo "=== Files to KEEP (open issues) ==="
if [[ ${#to_keep[@]} -eq 0 ]]; then
    echo "  (none)"
else
    for item in "${to_keep[@]}"; do
        echo "  ✓ $item"
    done
fi
echo ""

echo "=== Files to ARCHIVE (closed/not-found issues) ==="
if [[ ${#to_archive[@]} -eq 0 ]]; then
    echo "  (none)"
else
    for item in "${to_archive[@]}"; do
        IFS='|' read -r file num state <<< "$item"
        target="${ARCHIVE_DIR}/${DATE}-codex-${num}.md"
        echo "  → $file → $target (Issue #${num}: $state)"
    done
fi
echo ""

if $DRY_RUN; then
    echo "=== DRY-RUN complete ==="
    echo "Run with --apply to actually move ${#to_archive[@]} files"
else
    if [[ ${#to_archive[@]} -eq 0 ]]; then
        echo "Nothing to archive."
        exit 0
    fi
    
    echo "=== Archiving ${#to_archive[@]} files ==="
    for item in "${to_archive[@]}"; do
        IFS='|' read -r file num state <<< "$item"
        target="${ARCHIVE_DIR}/${DATE}-codex-${num}.md"
        mv "$file" "$target"
        echo "  ✓ Moved $file → $target"
    done
    echo ""
    echo "=== Done! Archived ${#to_archive[@]} files ==="
fi
