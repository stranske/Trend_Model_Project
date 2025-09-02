#!/bin/bash

# validate_fast.sh - Intelligent fast validation for Codex commits
# Automatically detects what type of validation is needed based on changes
# Usage: ./scripts/validate_fast.sh [--full] [--fix] [--verbose] [--commit-range=HEAD~1] [--profile]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
FULL_CHECK=false
FIX_MODE=false
VERBOSE_MODE=false
COMMIT_RANGE="HEAD~1"
PROFILE_MODE=false
START_TIME=$(date +%s)

# Parse arguments
for arg in "$@"; do
    case $arg in
        --full)
            FULL_CHECK=true
            ;;
        --fix)
            FIX_MODE=true
            ;;
        --verbose)
            VERBOSE_MODE=true
            ;;
        --commit-range=*)
            COMMIT_RANGE="${arg#*=}"
            ;;
        --profile)
            PROFILE_MODE=true
            ;;
    esac
done

# Profiling function
profile_step() {
    if [[ "$PROFILE_MODE" == true ]]; then
        local current_time=$(date +%s)
        local elapsed=$((current_time - START_TIME))
        echo -e "${MAGENTA}[${elapsed}s] $1${NC}"
    fi
}

echo -e "${CYAN}=== Intelligent Fast Validation ===${NC}"
profile_step "Starting validation"

# Activate virtual environment
if [[ -z "$VIRTUAL_ENV" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate > /dev/null 2>&1
    profile_step "Virtual environment activated"
fi

# Analyze what changed to determine optimal validation strategy
echo -e "${BLUE}Analyzing changes...${NC}"
CHANGED_FILES=$(git diff --name-only $COMMIT_RANGE 2>/dev/null | grep -v -E '^(Old/|notebooks/old/)' || echo "")
PYTHON_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(py)$' || true)
CONFIG_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(yml|yaml|toml|cfg|ini)$' || true)
TEST_FILES=$(echo "$PYTHON_FILES" | grep -E '^tests/' || true)
SRC_FILES=$(echo "$PYTHON_FILES" | grep -E '^src/' || true)
SCRIPT_FILES=$(echo "$PYTHON_FILES" | grep -E '^scripts/' || true)

# Count changes
TOTAL_PYTHON=$(echo "$PYTHON_FILES" | grep -v '^$' | wc -l || echo 0)
TOTAL_CONFIG=$(echo "$CONFIG_FILES" | grep -v '^$' | wc -l || echo 0)
TOTAL_TEST=$(echo "$TEST_FILES" | grep -v '^$' | wc -l || echo 0)
TOTAL_SRC=$(echo "$SRC_FILES" | grep -v '^$' | wc -l || echo 0)

echo -e "${BLUE}Change analysis:${NC}"
echo "  Python files: $TOTAL_PYTHON"
echo "  Config files: $TOTAL_CONFIG"
echo "  Test files: $TOTAL_TEST"
echo "  Source files: $TOTAL_SRC"

if [[ $TOTAL_PYTHON -eq 0 && $TOTAL_CONFIG -eq 0 ]]; then
    echo -e "${GREEN}✓ No Python or config changes detected - validation not needed${NC}"
    exit 0
fi

profile_step "Change analysis complete"

# Smart validation strategy
VALIDATION_STRATEGY="incremental"
if [[ "$FULL_CHECK" == true || $TOTAL_PYTHON -gt 10 ]]; then
    VALIDATION_STRATEGY="full"
elif [[ $TOTAL_SRC -gt 0 || $TOTAL_CONFIG -gt 0 ]]; then
    VALIDATION_STRATEGY="comprehensive"
fi

echo -e "${BLUE}Using $VALIDATION_STRATEGY validation strategy${NC}"
echo ""

# Validation functions
run_fast_check() {
    local name="$1"
    local command="$2"
    local fix_command="$3"
    local check_files="$4"
    
    echo -e "${BLUE}Checking $name...${NC}"
    
    # Use specific files if provided and not in full mode
    local actual_command="$command"
    if [[ -n "$check_files" && "$VALIDATION_STRATEGY" != "full" ]]; then
        actual_command=$(echo "$command" | sed "s|src/ tests/|$check_files|g")
    fi
    
    local start_check=$(date +%s)
    if eval "$actual_command" > /tmp/fast_check_output 2>&1; then
        local end_check=$(date +%s)
        local check_time=$((end_check - start_check))
        echo -e "${GREEN}✓ $name (${check_time}s)${NC}"
        return 0
    else
        echo -e "${RED}✗ $name${NC}"
        
        if [[ "$FIX_MODE" == true && -n "$fix_command" ]]; then
            echo -e "${YELLOW}  Auto-fixing...${NC}"
            if eval "$fix_command" > /tmp/fast_fix_output 2>&1; then
                if eval "$actual_command" > /tmp/fast_recheck_output 2>&1; then
                    echo -e "${GREEN}✓ $name (fixed)${NC}"
                    return 0
                fi
            fi
            echo -e "${RED}✗ $name (fix failed)${NC}"
        fi
        
        # Show first few lines of error
        echo -e "${YELLOW}  Error preview:${NC}"
        head -3 /tmp/fast_check_output | sed 's/^/    /'
        return 1
    fi
}

# Track validation results
FAILED_CHECKS=()
VALIDATION_SUCCESS=true

# Always check these basics (very fast)
echo -e "${CYAN}=== Basic Checks ===${NC}"

if ! run_fast_check "Import validation" "python -c 'import src.trend_analysis'" ""; then
    VALIDATION_SUCCESS=false
    FAILED_CHECKS+=("Import validation")
fi

profile_step "Import check complete"

# Formatting (always run, very fast to check)
if ! run_fast_check "Code formatting" "black --check ." "black ." "$PYTHON_FILES"; then
    VALIDATION_SUCCESS=false
    FAILED_CHECKS+=("Code formatting")
fi

profile_step "Formatting check complete"

# Syntax errors (critical, very fast)
if [[ -n "$PYTHON_FILES" ]]; then
    echo -e "${BLUE}Checking syntax...${NC}"
    SYNTAX_OK=true
    for file in $PYTHON_FILES; do
        if [[ -f "$file" ]]; then
            if ! python -m py_compile "$file" 2>/dev/null; then
                echo -e "${RED}✗ Syntax error in $file${NC}"
                SYNTAX_OK=false
                VALIDATION_SUCCESS=false
                FAILED_CHECKS+=("Syntax error in $file")
            fi
        fi
    done
    if [[ "$SYNTAX_OK" == true ]]; then
        echo -e "${GREEN}✓ Syntax check${NC}"
    fi
fi

profile_step "Syntax check complete"

# Strategy-based validation
case "$VALIDATION_STRATEGY" in
    "incremental")
        echo -e "${CYAN}=== Incremental Validation ===${NC}"
        
        # Only critical linting errors
        if ! run_fast_check "Critical linting" "flake8 --select=E9,F --statistics" "" "$PYTHON_FILES"; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Critical linting")
        fi
        
        # Basic type checking (only on a few files)
        if [[ $TOTAL_SRC -gt 0 ]]; then
            LIMITED_SRC=$(echo "$SRC_FILES" | head -3)
            if ! run_fast_check "Basic type check" "echo '$LIMITED_SRC' | xargs mypy --follow-imports=silent --ignore-missing-imports" "mypy --install-types --non-interactive"; then
                VALIDATION_SUCCESS=false
                FAILED_CHECKS+=("Basic type check")
            fi
        fi
        ;;
        
    "comprehensive")
        echo -e "${CYAN}=== Comprehensive Validation ===${NC}"
        
        # Full linting
        if ! run_fast_check "Full linting" "flake8 src/ tests/ scripts/ --statistics" ""; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Full linting")
        fi
        
        # Type checking
        if ! run_fast_check "Type checking" "mypy src/" "mypy --install-types --non-interactive"; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Type checking")
        fi
        
        # Quick test (only if test files changed)
        if [[ $TOTAL_TEST -gt 0 ]]; then
            # Use conditional verbosity for pytest
            if [[ "$VERBOSE_MODE" == true ]]; then
                PYTEST_VERBOSITY="-v --tb=short -x"
            else
                PYTEST_VERBOSITY="-q -x"
            fi
            
            if ! run_fast_check "Quick tests" "pytest $TEST_FILES $PYTEST_VERBOSITY" ""; then
                VALIDATION_SUCCESS=false
                FAILED_CHECKS+=("Quick tests")
            fi
        fi
        ;;
        
    "full")
        echo -e "${CYAN}=== Full Validation ===${NC}"
        echo -e "${YELLOW}Running comprehensive validation (may take longer)...${NC}"
        
        # All checks
        if ! run_fast_check "Full linting" "flake8 src/ tests/ scripts/" ""; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Full linting")
        fi
        
        if ! run_fast_check "Type checking" "mypy src/" "mypy --install-types --non-interactive"; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Type checking")
        fi
        
        # Use conditional verbosity for pytest
        if [[ "$VERBOSE_MODE" == true ]]; then
            PYTEST_VERBOSITY="-v --tb=short"
        else
            PYTEST_VERBOSITY="-q"
        fi
        
        if ! run_fast_check "All tests" "pytest tests/ $PYTEST_VERBOSITY" ""; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("All tests")
        fi
        
        if ! run_fast_check "Test coverage" "pytest --cov=src --cov-report=term-missing --cov-fail-under=70" ""; then
            VALIDATION_SUCCESS=false
            FAILED_CHECKS+=("Test coverage")
        fi
        ;;
esac

profile_step "Strategy validation complete"

# Final summary
echo ""
echo -e "${CYAN}=== Validation Summary ===${NC}"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

if [[ "$VALIDATION_SUCCESS" == true ]]; then
    echo -e "${GREEN}🎉 All validations passed in ${TOTAL_TIME}s!${NC}"
    echo -e "${GREEN}✓ Codex changes are ready for merge${NC}"
    
    if [[ "$VALIDATION_STRATEGY" == "incremental" ]]; then
        echo -e "${BLUE}ℹ  Run with --full for comprehensive validation${NC}"
    fi
    exit 0
else
    echo -e "${RED}❌ Validation failed in ${TOTAL_TIME}s${NC}"
    echo -e "${RED}Issues found:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  • $check${NC}"
    done
    echo ""
    echo -e "${YELLOW}Quick fixes:${NC}"
    echo "  • Run with --fix to auto-fix formatting/imports"
    echo "  • Use ./scripts/fix_common_issues.sh for common problems"
    echo "  • Use ./scripts/check_branch.sh --verbose for detailed output"
    echo "  • Run with --full for comprehensive validation"
    exit 1
fi
