#!/bin/bash
# Pre-commit hook to validate test dependencies
# Install: cp scripts/pre-commit-check-deps.sh .git/hooks/pre-commit

set -e

echo "🔍 Checking test dependencies..."

# Run the sync script in verify mode
if python scripts/sync_test_dependencies.py --verify 2>&1 | grep -q "undeclared"; then
    echo ""
    echo "❌ Test dependencies are not synchronized!"
    echo ""
    echo "Auto-fixing with: python scripts/sync_test_dependencies.py --fix"
    python scripts/sync_test_dependencies.py --fix
    echo ""
    echo "✅ Fixed! Please review and stage the changes to requirements.txt"
    echo ""
    echo "Run: git add requirements.txt"
    exit 1
fi

echo "✅ All test dependencies are declared"
exit 0
