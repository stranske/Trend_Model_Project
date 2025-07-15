#!/bin/bash

# dev_check.sh - Ultra-fast development validation for Codex workflow
# Usage: ./scripts/dev_check.sh [--fix] [--changed] [--verbose]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Command line options
FIX_MODE=false
CHANGED_ONLY=false
VERBOSE_MODE=false

for arg in "$@"; do
    case $arg in
        --fix)
            FIX_MODE=true
            ;;
        --changed)
            CHANGED_ONLY=true
            ;;
        --verbose)
            VERBOSE_MODE=true
            ;;
    esac
done

echo -e "${CYAN}=== Ultra-Fast Development Check ===${NC}"

# Activate virtual environment if needed
if [[ -z "$VIRTUAL_ENV" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate > /dev/null 2>&1
fi

# Determine files to check
if [[ "$CHANGED_ONLY" == true ]]; then
    # Only check files changed in the last commit or working directory
    PYTHON_FILES=$(git diff --name-only HEAD~1 2>/dev/null | grep -E '\.(py)$' | grep -v -E '^(Old/|notebooks/old/)' || true)
    UNSTAGED_FILES=$(git diff --name-only 2>/dev/null | grep -E '\.(py)$' | grep -v -E '^(Old/|notebooks/old/)' || true)
    ALL_FILES=$(echo -e "$PYTHON_FILES\n$UNSTAGED_FILES" | sort -u | grep -v '^$' || true)
    
    if [[ -z "$ALL_FILES" ]]; then
        echo -e "${GREEN}✓ No Python files changed (excluding old folders) - nothing to check${NC}"
        exit 0
    fi
    
    echo -e "${BLUE}Checking only changed files (excluding old folders):${NC}"
    echo "$ALL_FILES" | sed 's/^/  /'
    echo ""
else
    ALL_FILES="src/ tests/ scripts/"
fi

# Function to run quick checks
quick_check() {
    local name="$1"
    local command="$2"
    local fix_command="$3"
    
    if [[ "$VERBOSE_MODE" == true ]]; then
        echo -e "${BLUE}Running: $command${NC}"
    fi
    
    if eval "$command" > /tmp/quick_check_output 2>&1; then
        echo -e "${GREEN}✓ $name${NC}"
        return 0
    else
        echo -e "${RED}✗ $name${NC}"
        
        if [[ "$FIX_MODE" == true && -n "$fix_command" ]]; then
            echo -e "${YELLOW}  Fixing...${NC}"
            if eval "$fix_command" > /tmp/quick_fix_output 2>&1; then
                # Re-check
                if eval "$command" > /tmp/quick_recheck_output 2>&1; then
                    echo -e "${GREEN}✓ $name (fixed)${NC}"
                    return 0
                fi
            fi
            echo -e "${RED}✗ $name (fix failed)${NC}"
        fi
        
        if [[ "$VERBOSE_MODE" == true ]]; then
            echo -e "${YELLOW}Output:${NC}"
            head -10 /tmp/quick_check_output | sed 's/^/  /'
        fi
        return 1
    fi
}

# Quick syntax check first (fastest)
echo -e "${BLUE}1. Syntax check...${NC}"
if [[ "$CHANGED_ONLY" == true && -n "$ALL_FILES" ]]; then
    SYNTAX_OK=true
    for file in $ALL_FILES; do
        if [[ -f "$file" ]]; then
            if ! python -m py_compile "$file" 2>/dev/null; then
                echo -e "${RED}✗ Syntax error in $file${NC}"
                SYNTAX_OK=false
            fi
        fi
    done
    if [[ "$SYNTAX_OK" == true ]]; then
        echo -e "${GREEN}✓ Syntax check${NC}"
    fi
else
    quick_check "Syntax check" "python -m compileall src/ -q" ""
fi

# Quick import test
echo -e "${BLUE}2. Import test...${NC}"
quick_check "Import test" "python -c 'import src.trend_analysis' 2>/dev/null" ""

# Formatting check (very fast)
echo -e "${BLUE}3. Formatting...${NC}"
if [[ "$CHANGED_ONLY" == true && -n "$ALL_FILES" ]]; then
    FMT_CMD="echo '$ALL_FILES' | xargs black --check"
    FIX_CMD="echo '$ALL_FILES' | xargs black"
else
    FMT_CMD="black --check ."
    FIX_CMD="black ."
fi
quick_check "Black formatting" "$FMT_CMD" "$FIX_CMD"

# Basic linting (only critical issues)
echo -e "${BLUE}4. Critical linting...${NC}"
if [[ "$CHANGED_ONLY" == true && -n "$ALL_FILES" ]]; then
    # Only check for critical errors (E9**, F***)
    LINT_CMD="echo '$ALL_FILES' | xargs flake8 --select=E9,F --statistics"
else
    LINT_CMD="flake8 src/ tests/ scripts/ --select=E9,F --statistics"
fi
quick_check "Critical lint errors" "$LINT_CMD" ""

# Type hints (basic check)
echo -e "${BLUE}5. Basic type check...${NC}"
if [[ "$CHANGED_ONLY" == true && -n "$ALL_FILES" ]]; then
    # For changed files only, run a lighter mypy check
    TYPE_CMD="echo '$ALL_FILES' | head -3 | xargs mypy --follow-imports=silent --ignore-missing-imports"
else
    TYPE_CMD="mypy src/ --follow-imports=silent --ignore-missing-imports"
fi
quick_check "Basic type check" "$TYPE_CMD" "mypy --install-types --non-interactive"

echo ""
echo -e "${CYAN}=== Quick Check Complete ===${NC}"
echo -e "${BLUE}For comprehensive validation, run: ./scripts/check_branch.sh${NC}"
echo -e "${BLUE}For auto-fixes, run with: --fix${NC}"
echo -e "${BLUE}For changed files only: --changed${NC}"
