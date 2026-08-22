#!/bin/bash

# quick_check.sh - Fast quality checks for development
# Usage: ./scripts/quick_check.sh

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Quick Branch Check ===${NC}"

# Activate virtual environment if needed
if [[ -z "$VIRTUAL_ENV" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Determine changed Python files (latest commit + working tree)
if ! DIFF_FILES=$(git diff --name-only HEAD~1 2>/dev/null); then
    echo "::warning::git diff command failed, but continuing. Recent changes check may be incomplete."
    CHANGED_FILES=""
else
    CHANGED_FILES=""
    while IFS= read -r changed_file; do
        [[ -n "$changed_file" ]] || continue
        [[ "$changed_file" == *.py ]] || continue
        [[ "$changed_file" =~ ^(Old/|notebooks/old/|archives/legacy_assets/) ]] && continue
        CHANGED_FILES+="${changed_file}"$'\n'
        if [[ $(printf '%s' "$CHANGED_FILES" | wc -l | tr -d ' ') -ge 5 ]]; then
            break
        fi
    done <<< "$DIFF_FILES"
    CHANGED_FILES="${CHANGED_FILES%$'\n'}"
fi
if [[ -n "$CHANGED_FILES" ]]; then
    CHANGED_FILES_ARRAY=()
    while IFS= read -r changed_file; do
        [[ -n "$changed_file" ]] || continue
        CHANGED_FILES_ARRAY[${#CHANGED_FILES_ARRAY[@]}]="$changed_file"
    done <<< "$CHANGED_FILES"
else
    CHANGED_FILES_ARRAY=()
fi

run_with_timeout() {
    local timeout_seconds="$1"
    shift
    python - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

try:
    completed = subprocess.run(sys.argv[2:], check=False, timeout=float(sys.argv[1]))
except subprocess.TimeoutExpired:
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
}

CHECK_STATUS=0

# Quick format check
echo -e "${BLUE}Checking formatting...${NC}"
if [[ ${#CHANGED_FILES_ARRAY[@]} -gt 0 ]]; then
    if run_with_timeout 30 black --check "${CHANGED_FILES_ARRAY[@]}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Formatting OK${NC}"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            echo -e "${RED}✗ Formatting check timed out${NC}"
        else
            echo -e "${RED}✗ Formatting issues (run: black ${CHANGED_FILES_ARRAY[*]})${NC}"
        fi
        CHECK_STATUS=1
    fi
else
    echo -e "${GREEN}✓ No Python files changed (skipping formatting check)${NC}"
fi

# Quick lint check on recent changes
echo -e "${BLUE}Checking recent changes...${NC}"
if [[ ${#CHANGED_FILES_ARRAY[@]} -gt 0 ]]; then
    if run_with_timeout 30 flake8 "${CHANGED_FILES_ARRAY[@]}" 2>/dev/null; then
        echo -e "${GREEN}✓ Recent changes look good${NC}"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            echo -e "${RED}✗ Linting check timed out${NC}"
        else
            echo -e "${RED}✗ Linting issues in recent changes${NC}"
        fi
        CHECK_STATUS=1
    fi
else
    echo -e "${GREEN}✓ No Python files changed (excluding old folders)${NC}"
fi

# Quick import test
echo -e "${BLUE}Testing imports...${NC}"
if python -c "import src.trend_analysis" 2>/dev/null; then
    echo -e "${GREEN}✓ Package imports successfully${NC}"
else
    echo -e "${RED}✗ Import errors${NC}"
    CHECK_STATUS=1
fi

echo -e "${BLUE}Quick check complete. Run ./scripts/check_branch.sh for full validation${NC}"
exit "$CHECK_STATUS"
