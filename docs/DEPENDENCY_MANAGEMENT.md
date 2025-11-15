# Test Dependency Management

This document describes the comprehensive dependency validation system for the Trend Analysis Project test suite.

> **Update (2025-10):** Dependency declarations are now maintained in
> `pyproject.toml` (including the dev extra) with pins captured in
> `requirements.lock`. Historical references to `requirements.txt` in this
> document refer to the legacy workflow.

# Test Dependency Management

## System Status: ENFORCED ✅

As of 2025-10-28, this project implements **automatic dependency enforcement** that prevents tests from running with missing dependencies.

**Current Status**: 2,102 tests pass, 0 skipped

### Key Features

1. **Automatic Installation**: CI automatically installs all required dependencies (Python packages, Node.js, npm, uv)
2. **Validation Tests**: Test suite validates all dependencies are present before running
3. **Enforcement Tests**: Build fails if new dependencies are used without being declared
4. **Zero Skipped Tests**: All tests must run; skipping due to missing dependencies is not allowed

See [DEPENDENCY_ENFORCEMENT.md](DEPENDENCY_ENFORCEMENT.md) for complete implementation details.

---

## Overview

The project uses a multi-layered approach to ensure all test dependencies are available:

1. **Automated CI validation** - Dependencies checked automatically in CI workflows
2. **Test suite validation** - Comprehensive test suite validates all dependencies
3. **Manual check script** - Quick command-line tool for local validation
4. **Configuration files** - Explicit dependency declarations with documentation

## Quick Start

### Check Dependencies Locally

```bash
# Quick check with color-coded output
./scripts/check_test_dependencies.sh

# Run dependency validation tests
pytest tests/test_test_dependencies.py -v
```

### Install Missing Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install with dev dependencies
pip install -e '.[dev]'

# External tools (must be installed separately)
# - Node.js v20+: https://nodejs.org/
# - uv: https://github.com/astral-sh/uv
```

## Dependency Categories

### Required Python Packages

These packages are **required** for the test suite to run. Tests will fail if these are missing:

- **Python 3.11+** (enforced minimum version)
- **pytest** >= 8.0 - Test framework
- **coverage** >= 7.0 - Coverage measurement
- **hypothesis** >= 6.0 - Property-based testing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **pydantic** - Data validation
- **PyYAML** - Configuration parsing
- **requests** - HTTP library
- **jsonschema** - JSON validation
- **streamlit** - Web application framework
- **fastapi** - API framework
- **httpx** >= 0.25 - Async HTTP client

### Optional Python Packages

These packages are **optional**. Tests requiring them will skip gracefully if missing:

- **black** - Code formatting
- **ruff** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks

### External CLI Tools

These tools are **not pip-installable** and must be installed separately. Tests requiring them will skip with clear messages:

- **Node.js v20+** and **npm** - Required for JavaScript workflow tests
  - 19 tests skip without Node.js
  - Install from: https://nodejs.org/
  
- **uv** - Optional, for lockfile consistency tests
  - 1 test skips without uv
  - Install from: https://github.com/astral-sh/uv

## Configuration Files

### requirements.txt

Primary dependency file with explicit test dependencies section:

```txt
# Testing dependencies
pytest>=8.0
pytest-cov
pytest-rerunfailures
hypothesis
coverage
jsonschema

# Note: Node.js (v20+) and npm are required for JavaScript workflow tests
# Install from: https://nodejs.org/
# Optional: uv (for lockfile tests) - https://github.com/astral-sh/uv
```

### pyproject.toml

Build configuration with optional dev dependencies:

```toml
[project.optional-dependencies]
dev = [
    "pytest==8.4.2",
    "pytest-cov==7.0.0",
    "pytest-rerunfailures>=13.0",
    "pytest-xdist",  # parallel test execution
    "coverage>=7.0",
    "hypothesis>=6.0",
    # ... other dev tools
]

# Note: External dependencies not managed by pip:
# - Node.js (v20+) and npm - Required for JavaScript workflow tests
# - uv - Optional, for lockfile consistency tests
```

## Validation Tools

### 1. CI Workflow Validation

**File**: `.github/workflows/reusable-10-ci-python.yml`

Automatically runs after installing dependencies in all CI jobs:

```yaml
- name: Validate test dependencies
  run: |
    # Runs check_test_dependencies.sh if available
    # Falls back to basic validation
    # Outputs to GitHub step summary
```

**Behavior**:
- ✅ Runs automatically on every CI build
- ✅ Reports results in GitHub Actions step summary
- ✅ Documents available and missing dependencies
- ✅ Does not fail build on missing optional dependencies

### 2. Test Suite Validation

**File**: `tests/test_test_dependencies.py`

Comprehensive pytest test suite with 13+ validation tests:

```python
class TestDependencies:
    def test_python_version()
    def test_required_packages_importable()
    def test_optional_packages_documented()
    def test_node_available()
    def test_npm_available_if_node_present()
    def test_uv_availability_documented()
    def test_requirements_file_exists()
    def test_pytest_plugins_available()
    def test_coverage_tool_available()
    def test_github_scripts_dependencies()
    def test_streamlit_dependencies()
    # ... additional tests
```

**Run with**:
```bash
pytest tests/test_test_dependencies.py -v
```

**Behavior**:
- ✅ Fails if required dependencies missing
- ✅ Skips gracefully if optional dependencies missing
- ✅ Provides installation instructions in skip messages
- ✅ Validates both Python packages and CLI tools

### 3. Manual Check Script

**File**: `scripts/check_test_dependencies.sh`

Quick command-line tool for local validation:

```bash
./scripts/check_test_dependencies.sh
```

**Features**:
- ✅ Color-coded output (green ✓, red ✗, yellow ○)
- ✅ Checks Python version
- ✅ Validates all required packages
- ✅ Checks optional packages
- ✅ Validates CLI tools (node, npm, uv, coverage)
- ✅ Provides installation instructions
- ✅ Exit code 0 if all required present, 1 otherwise

**Example output**:
```
=== Test Dependencies Check ===

Checking Python version...
✓ Python 3.11.14 (>=3.11 required)

Checking required Python packages...
✓ pytest
✓ coverage
✓ hypothesis
...

Checking optional Python packages...
✓ black
○ pre-commit (not found)

Checking Node.js...
○ Node.js (not found - JavaScript tests will be skipped)
  Install from: https://nodejs.org/

=== Summary ===
All required dependencies are available!
```

## Test Behavior with Missing Dependencies

### Required Dependencies Missing

If a **required** dependency is missing:
- ❌ Test suite fails with clear error
- ❌ CI build fails
- ❌ Manual script exits with code 1

### Optional Dependencies Missing

If an **optional** dependency is missing:
- ✅ Test suite runs
- ⏭️ Tests requiring the dependency skip with message
- ✅ CI build succeeds
- ℹ️ Manual script documents availability

**Example skip messages**:

```
SKIPPED [19] Node.js not found in PATH. JavaScript workflow tests will be skipped.
Install Node.js: https://nodejs.org/

SKIPPED [1] uv not found in PATH. Lockfile consistency tests will be skipped.
Install uv: https://github.com/astral-sh/uv

SKIPPED [1] Optional packages not available: pre-commit
Install with: pip install pre-commit
```

## Updating Dependencies

### Adding New Required Dependencies

1. Add to `requirements.txt` under the testing section
2. Add to `pyproject.toml` under `[project.optional-dependencies] dev`
3. Add import check in `test_test_dependencies.py::test_required_packages_importable()`
4. Update `scripts/check_test_dependencies.sh` required packages list
5. Update this document

### Adding New Optional Dependencies

1. Add to `pyproject.toml` under `[project.optional-dependencies] dev`
2. Add skip decorator to tests requiring it: `@pytest.mark.skipif(...)`
3. Add to `test_test_dependencies.py::test_optional_packages_documented()`
4. Add to `scripts/check_test_dependencies.sh` optional packages list
5. Document in this file

### Adding External CLI Tools

1. Add skip logic to tests requiring it
2. Document in configuration file comments
3. Add check to `test_test_dependencies.py`
4. Add to `scripts/check_test_dependencies.sh`
5. Update this document with installation instructions

## Troubleshooting

### "Module not found" errors

```bash
# Install all dependencies
pip install -r requirements.txt

# Or with dev dependencies
pip install -e '.[dev]'
```

### JavaScript tests skipping

```bash
# Install Node.js v20+ from https://nodejs.org/
# Verify installation
node --version
npm --version

# Re-run tests
pytest -v
```

### Lockfile tests skipping

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# Re-run tests
pytest tests/ -k lockfile -v
```

### CI validation not running

1. Check that `reusable-10-ci-python.yml` has the "Validate test dependencies" step
2. Check that `scripts/check_test_dependencies.sh` is executable (`chmod +x`)
3. Check CI logs for step summary output

## Integration with CI/CD

The dependency validation is integrated into the CI pipeline as follows:

1. **Gate Workflow** (`.github/workflows/pr-00-gate.yml`)
   - Calls `reusable-10-ci-python.yml` for all Python testing
   
2. **Reusable Python CI** (`.github/workflows/reusable-10-ci-python.yml`)
   - Installs dependencies
   - **→ Validates test dependencies** (new step)
   - Runs linting, type checking
   - Runs test suite (which includes dependency tests)
   - Checks coverage

3. **Test Suite** (`tests/test_test_dependencies.py`)
   - Runs as part of normal test execution
   - Validates environment programmatically
   - Provides detailed skip messages

## Design Philosophy

The dependency management system follows these principles:

1. **Fail Fast**: Required dependencies fail the build immediately
2. **Graceful Degradation**: Optional dependencies allow tests to skip
3. **Clear Communication**: Skip messages include installation instructions
4. **Multiple Validation Layers**: CI, test suite, and manual checks
5. **Developer-Friendly**: Color-coded output and helpful messages
6. **CI-Friendly**: Outputs to step summaries for visibility
7. **Documentation**: Inline comments in configuration files

## Related Documentation

- [Testing Guide](TESTING_SUMMARY.md) - Overview of test infrastructure
- [Coverage Guide](coverage-summary.md) - Coverage tracking and thresholds
- [GitHub Copilot Instructions](../.github/copilot-instructions.md) - Development workflow
