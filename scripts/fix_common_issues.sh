#!/bin/bash

# fix_common_issues.sh - Quickly fix common code quality issues
# Usage: ./scripts/fix_common_issues.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Quick Fix Common Issues ===${NC}"

# Activate virtual environment if needed
if [[ -z "$VIRTUAL_ENV" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
fi

# 1. Fix formatting with Black
echo -e "${BLUE}Fixing code formatting...${NC}"
if black . > /tmp/black_output 2>&1; then
    echo -e "${GREEN}✓ Code formatting fixed${NC}"
else
    echo -e "${RED}✗ Black formatting failed${NC}"
    cat /tmp/black_output
fi

# 2. Try to fix basic MyPy issues
echo -e "${BLUE}Installing missing type stubs...${NC}"
if mypy --install-types --non-interactive > /tmp/mypy_install 2>&1; then
    echo -e "${GREEN}✓ Type stubs installed${NC}"
else
    echo -e "${YELLOW}⚠ Some type stubs couldn't be installed${NC}"
fi

# 3. Stubs now cover yaml/requests, so no blanket ignores are added
echo -e "${BLUE}Type ignore insertion skipped (stubs installed)${NC}"

# 4. Fix simple line length issues (only for very long lines)
echo -e "${BLUE}Fixing very long lines...${NC}"
find src/ tests/ scripts/ -name "*.py" -exec grep -l ".\{120,\}" {} \; | while read file; do
    # Only fix extremely long lines (120+ chars) to avoid breaking code
    if grep -q ".\{120,\}" "$file"; then
        echo "  Found very long lines in $file (manual review needed)"
    fi
done

# 5. Check results
echo ""
echo -e "${BLUE}=== Quick Fix Results ===${NC}"

# Run quick checks
echo -e "${BLUE}Black formatting:${NC}"
if black --check . > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo -e "${BLUE}MyPy type checking:${NC}"
if mypy src/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo -e "${BLUE}Basic import test:${NC}"
if python -c "import src.trend_analysis" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo ""
echo -e "${GREEN}Quick fixes complete! Run ./scripts/check_branch.sh for full validation${NC}"
