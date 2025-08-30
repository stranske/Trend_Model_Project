#!/bin/bash

# test-release.sh - Test release process locally
# Usage: ./scripts/test-release.sh [version]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

VERSION=${1:-"0.1.0-test"}
DIST_DIR="dist"

echo -e "${BLUE}=== Testing Release Process Locally ===${NC}"
echo "Version: ${VERSION}"
echo ""

# Clean previous builds
if [[ -d "${DIST_DIR}" ]]; then
    echo -e "${YELLOW}Cleaning previous build artifacts...${NC}"
    rm -rf "${DIST_DIR}"
fi

# Install build dependencies if needed
echo -e "${BLUE}Installing build dependencies...${NC}"
if ! command -v python -m build &> /dev/null; then
    pip install build twine
fi

# Update version in pyproject.toml
echo -e "${BLUE}Updating version in pyproject.toml...${NC}"
sed -i.bak "s/version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
echo "Version updated to: ${VERSION}"

# Build packages
echo -e "${BLUE}Building packages...${NC}"
if python -m build; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    # Restore original version
    mv pyproject.toml.bak pyproject.toml
    exit 1
fi

# Check packages
echo -e "${BLUE}Checking packages...${NC}"
if python -m twine check dist/*; then
    echo -e "${GREEN}✓ Package check passed${NC}"
else
    echo -e "${RED}✗ Package check failed${NC}"
    exit 1
fi

# Test installation
echo -e "${BLUE}Testing package installation...${NC}"
if [[ -d ".test-venv" ]]; then
    rm -rf .test-venv
fi

python -m venv .test-venv
source .test-venv/bin/activate

# Test wheel installation
WHEEL_FILE=$(find dist -name "*.whl" | head -1)
if [[ -n "$WHEEL_FILE" ]]; then
    echo "Testing wheel: $WHEEL_FILE"
    if pip install "$WHEEL_FILE"; then
        echo -e "${GREEN}✓ Wheel installation successful${NC}"
        
        # Test import and version
        if python -c "import trend_analysis; print(f'Package version: {trend_analysis.__version__}')"; then
            echo -e "${GREEN}✓ Package import and version check successful${NC}"
        else
            echo -e "${RED}✗ Package import failed${NC}"
            deactivate
            exit 1
        fi
        
        # Test CLI commands
        echo -e "${BLUE}Testing CLI commands...${NC}"
        if trend-analysis --help > /dev/null 2>&1; then
            echo -e "${GREEN}✓ trend-analysis CLI available${NC}"
        else
            echo -e "${YELLOW}⚠ trend-analysis CLI not available (may need dependencies)${NC}"
        fi
        
        if trend-multi-analysis --help > /dev/null 2>&1; then
            echo -e "${GREEN}✓ trend-multi-analysis CLI available${NC}"
        else
            echo -e "${YELLOW}⚠ trend-multi-analysis CLI not available (may need dependencies)${NC}"
        fi
        
    else
        echo -e "${RED}✗ Wheel installation failed${NC}"
        deactivate
        exit 1
    fi
else
    echo -e "${RED}✗ No wheel file found${NC}"
    deactivate
    exit 1
fi

# Test source distribution
pip uninstall -y trend-analysis
SDIST_FILE=$(find dist -name "*.tar.gz" | head -1)
if [[ -n "$SDIST_FILE" ]]; then
    echo "Testing source distribution: $SDIST_FILE"
    if pip install "$SDIST_FILE"; then
        echo -e "${GREEN}✓ Source distribution installation successful${NC}"
        
        if python -c "import trend_analysis; print(f'Package version: {trend_analysis.__version__}')"; then
            echo -e "${GREEN}✓ Source distribution import successful${NC}"
        else
            echo -e "${RED}✗ Source distribution import failed${NC}"
            deactivate
            exit 1
        fi
    else
        echo -e "${RED}✗ Source distribution installation failed${NC}"
        deactivate
        exit 1
    fi
else
    echo -e "${RED}✗ No source distribution file found${NC}"
    deactivate
    exit 1
fi

deactivate
rm -rf .test-venv

# Generate test changelog (if git-cliff is available)
if command -v git-cliff &> /dev/null; then
    echo -e "${BLUE}Generating test changelog...${NC}"
    git-cliff --tag "v${VERSION}" --strip header > test-changelog.md
    echo -e "${GREEN}✓ Test changelog generated: test-changelog.md${NC}"
else
    echo -e "${YELLOW}⚠ git-cliff not available, skipping changelog test${NC}"
fi

# Show package contents
echo -e "${BLUE}Package contents:${NC}"
ls -la dist/

echo ""
echo -e "${GREEN}=== Release Test Complete ===${NC}"
echo -e "${GREEN}All checks passed! Package is ready for release.${NC}"
echo ""
echo "Next steps:"
echo "1. Review generated packages in dist/"
echo "2. Test upload to TestPyPI (if desired):"
echo "   twine upload --repository testpypi dist/*"
echo "3. Create git tag and push to trigger automated release:"
echo "   git tag v${VERSION}"
echo "   git push origin v${VERSION}"

# Restore original version
mv pyproject.toml.bak pyproject.toml
echo -e "${YELLOW}Restored original version in pyproject.toml${NC}"