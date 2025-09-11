#!/bin/bash

# git_hooks.sh - Set up Git hooks for automatic Codex validation
# Usage: ./scripts/git_hooks.sh [install|uninstall|status]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

GIT_HOOKS_DIR=".git/hooks"
PROJECT_ROOT=$(git rev-parse --show-toplevel)

show_usage() {
    echo "Usage: $0 [install|uninstall|status]"
    echo ""
    echo "Commands:"
    echo "  install   - Install Git hooks for automatic validation"
    echo "  uninstall - Remove installed Git hooks"
    echo "  status    - Show current hook status"
    echo ""
    echo "Hooks that will be installed:"
    echo "  pre-commit    - Fast validation before commits"
    echo "  pre-push      - Comprehensive validation before pushes"
    echo "  post-commit   - Optional notification after commits"
}

install_hooks() {
    echo -e "${CYAN}Installing Git hooks for Codex validation...${NC}"
    
    # Create hooks directory if it doesn't exist
    mkdir -p "$GIT_HOOKS_DIR"
    
    # Pre-commit hook (fast validation with auto-debugging)
    cat > "$GIT_HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook: Fast validation for Codex commits

set -e

echo "🔍 Running pre-commit validation..."

# Check if we're in the right directory
if [[ ! -f "scripts/validate_fast.sh" ]]; then
    echo "
    Validation script not found, skipping pre-commit checks"
    exit 0
fi

# Run fast validation on staged changes
if ! ./scripts/validate_fast.sh --commit-range=HEAD; then
    echo ""
    echo "❌ Initial pre-commit validation failed"
    echo "🔧 Attempting automatic fixes..."
    if ./scripts/fix_common_issues.sh > "$HOME/.pre_commit_autofix.log" 2>&1; then
        echo "♻️  Re-running validation after fixes..."
        if ./scripts/validate_fast.sh --commit-range=HEAD; then
            echo "✅ Validation passed after automatic fixes!"
            exit 0
        fi
    fi
    echo ""
    echo "❌ Pre-commit validation failed!"
    echo "💡 Fix issues or use 'git commit --no-verify' to skip checks"
    echo "🔧 Manual fixes: ./scripts/validate_fast.sh --fix"
    exit 1
fi

echo "✅ Pre-commit validation passed!"
EOF
    
    # Pre-push hook (comprehensive validation)
    cat > "$GIT_HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Pre-push hook: Comprehensive validation before pushing

set -e

echo "🚀 Running pre-push validation..."

# Check if we're in the right directory
if [[ ! -f "scripts/check_branch.sh" ]]; then
    echo "⚠️  Validation script not found, skipping pre-push checks"
    exit 0
fi

# Run comprehensive validation
if ! ./scripts/check_branch.sh --fast; then
    echo ""
    echo "❌ Pre-push validation failed!"
    echo "💡 Fix issues or use 'git push --no-verify' to skip checks"
    echo "🔧 Try: ./scripts/check_branch.sh --fix --verbose"
    exit 1
fi

echo "✅ Pre-push validation passed!"
EOF
    
    # Post-commit hook (optional notification)
    cat > "$GIT_HOOKS_DIR/post-commit" << 'EOF'
#!/bin/bash
# Post-commit hook: Notification and quick status

# Get commit info
COMMIT_HASH=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
COMMIT_AUTHOR=$(git log -1 --pretty=%an)

echo ""
echo "📝 Commit successful: $COMMIT_HASH"
echo "👤 Author: $COMMIT_AUTHOR"
echo "💬 Message: $COMMIT_MSG"

# Quick status check
if [[ -f "scripts/dev_check.sh" ]]; then
    echo ""
    echo "🔍 Running post-commit quick check..."
    if ./scripts/dev_check.sh --changed > /dev/null 2>&1; then
        echo "✅ Quick check passed"
    else
        echo "⚠️  Quick check found issues - consider running validation"
        echo "🔧 Run: ./scripts/validate_fast.sh --fix"
    fi
fi
EOF
    
    # Make hooks executable
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-push"
    chmod +x "$GIT_HOOKS_DIR/post-commit"
    
    echo -e "${GREEN}✅ Git hooks installed successfully!${NC}"
    echo ""
    echo "Installed hooks:"
    echo "  ✓ pre-commit  - Fast validation before commits"
    echo "  ✓ pre-push    - Comprehensive validation before pushes"
    echo "  ✓ post-commit - Status notification after commits"
    echo ""
    echo "To bypass hooks temporarily:"
    echo "  git commit --no-verify"
    echo "  git push --no-verify"
}

uninstall_hooks() {
    echo -e "${YELLOW}Uninstalling Git hooks...${NC}"
    
    HOOKS=("pre-commit" "pre-push" "post-commit")
    REMOVED=0
    
    for hook in "${HOOKS[@]}"; do
        if [[ -f "$GIT_HOOKS_DIR/$hook" ]]; then
            rm "$GIT_HOOKS_DIR/$hook"
            echo "  ✓ Removed $hook"
            REMOVED=$((REMOVED + 1))
        fi
    done
    
    if [[ $REMOVED -gt 0 ]]; then
        echo -e "${GREEN}✅ Removed $REMOVED Git hook(s)${NC}"
    else
        echo -e "${BLUE}ℹ️  No Git hooks found to remove${NC}"
    fi
}

show_status() {
    echo -e "${CYAN}Git Hooks Status${NC}"
    echo ""
    
    HOOKS=("pre-commit" "pre-push" "post-commit")
    INSTALLED=0
    
    for hook in "${HOOKS[@]}"; do
        if [[ -f "$GIT_HOOKS_DIR/$hook" ]]; then
            echo -e "  ✅ $hook - installed"
            INSTALLED=$((INSTALLED + 1))
        else
            echo -e "  ❌ $hook - not installed"
        fi
    done
    
    echo ""
    if [[ $INSTALLED -eq 3 ]]; then
        echo -e "${GREEN}All validation hooks are installed${NC}"
    elif [[ $INSTALLED -gt 0 ]]; then
        echo -e "${YELLOW}Some hooks are installed ($INSTALLED/3)${NC}"
    else
        echo -e "${RED}No validation hooks installed${NC}"
        echo "Run: $0 install"
    fi
    
    # Check if validation scripts exist
    echo ""
    echo "Required scripts:"
    if [[ -f "scripts/validate_fast.sh" ]]; then
        echo "  ✅ validate_fast.sh - available"
    else
        echo "  ❌ validate_fast.sh - missing"
    fi
    
    if [[ -f "scripts/check_branch.sh" ]]; then
        echo "  ✅ check_branch.sh - available"
    else
        echo "  ❌ check_branch.sh - missing"
    fi
    
    if [[ -f "scripts/dev_check.sh" ]]; then
        echo "  ✅ dev_check.sh - available"
    else
        echo "  ❌ dev_check.sh - missing"
    fi
}

# Main command processing
case "${1:-status}" in
    install)
        install_hooks
        ;;
    uninstall)
        uninstall_hooks
        ;;
    status)
        show_status
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
