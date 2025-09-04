#!/bin/bash

# quick_check.sh - Fast quality checks for development
# Usage: ./scripts/quick_check.sh

set -e

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

# Quick format check
echo -e "${BLUE}Checking formatting...${NC}"
if black --check . > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Formatting OK${NC}"
else
    echo -e "${RED}✗ Formatting issues (run: black .)${NC}"
fi

# Quick lint check on recent changes
echo -e "${BLUE}Checking recent changes...${NC}"
CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null | grep -E '\.(py)$' 2>/dev/null | grep -v -E '^(Old/|notebooks/old/)' 2>/dev/null | head -5)
if [[ $? -ne 0 ]]; then
    echo "::warning::git diff command failed, but continuing. Recent changes check may be incomplete."
    CHANGED_FILES=""
fi
if [[ -n "$CHANGED_FILES" ]]; then
    if echo "$CHANGED_FILES" | xargs flake8 2>/dev/null; then
        echo -e "${GREEN}✓ Recent changes look good${NC}"
    else
        echo -e "${RED}✗ Linting issues in recent changes${NC}"
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
fi

echo -e "${BLUE}Quick check complete. Run ./scripts/check_branch.sh for full validation${NC}"
