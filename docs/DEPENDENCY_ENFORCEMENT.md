# Dependency Enforcement System

## Overview

This project implements a comprehensive dependency enforcement system to ensure that all tests can run without skipping due to missing dependencies. The system automatically validates, documents, and enforces dependency requirements across the test suite.

> **Update (2025-10):** The authoritative dependency list now lives in
> `pyproject.toml` (base + optional extras) with a generated
> `requirements.lock`. Any legacy references to `requirements.txt` in this
> document should be read as updates to the dev extra in `pyproject.toml`
> followed by `make lock`.

## System Components

### 1. Dependency Installation (CI)

**File**: `.github/workflows/reusable-10-ci-python.yml`

The CI workflow now automatically installs all required dependencies before running tests:

```yaml
- name: Set up Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '20'

- name: Install uv
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "$HOME/.local/bin" >> $GITHUB_PATH
```

### 2. Dependency Validation Tests

**File**: `tests/test_test_dependencies.py`

Validates that all required dependencies are present and accessible:

- **Python version**: Ensures Python 3.11+ is installed
- **Required packages**: Tests that core dependencies (pytest, pandas, numpy, etc.) are importable
- **Optional packages**: Documents which optional packages (black, ruff, mypy, pre_commit) are available
- **External tools**: Validates Node.js, npm, and uv are installed and functional

### 3. Dependency Enforcement Tests

**File**: `tests/test_dependency_enforcement.py`

**Critical Feature**: These tests FAIL if a new dependency is added to test code without being declared in `pyproject.toml`.

#### What It Does

1. **Scans all test files** for import statements
2. **Extracts top-level module names** from imports
3. **Compares against declared dependencies** in `pyproject.toml`
4. **Fails the build** if undeclared dependencies are found

#### Example Failure

If you add this to a test file:

```python
import some_new_package
```

Without adding `some_new_package` to `[project.optional-dependencies].dev`, the test will fail with:

```
Failed: The following imports in test files are not declared in pyproject.toml:
some_new_package

Add these packages to `[project.optional-dependencies].dev` to ensure tests can run.

🔧 To automatically fix this, run:
   python scripts/sync_test_dependencies.py --fix
   uv pip compile pyproject.toml -o requirements.lock
```

### 4. Automatic Dependency Synchronization

**File**: `scripts/sync_test_dependencies.py`

**The Missing Link**: This script automatically fixes dependency issues detected by the enforcement tests.

#### Usage

```bash
# Dry run - show what would change
python scripts/sync_test_dependencies.py

# Fix pyproject.toml automatically
python scripts/sync_test_dependencies.py --fix

# Verify mode - exit 1 if changes needed (for CI)
python scripts/sync_test_dependencies.py --verify

# Confirm formatter/test tool pins match the canonical versions
python -m scripts.sync_tool_versions --check
```

#### What It Does

1. Scans all test files for imports
2. Identifies which imports are not in `pyproject.toml`
3. Automatically adds missing packages to the dev extra
4. Outputs instructions to regenerate `requirements.lock`

#### CI Integration

When the enforcement test fails in CI:
1. `check-deps` step runs the sync script in verify mode
2. If it fails, the `auto-fix` step runs `--fix` mode
3. The updated `pyproject.toml` is shown in the job summary
4. **The build still fails** to force the developer to commit the changes

This ensures:
- Dependencies are never forgotten
- Changes are explicit in git history
- Manual review before merging

### 5. Pre-commit Hook (Optional)

**File**: `scripts/pre-commit-check-deps.sh`

Developers can install a local pre-commit hook that automatically syncs dependencies:

```bash
# Install the hook
cp scripts/pre-commit-check-deps.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Now every commit will:
# 1. Check if test dependencies are synchronized
# 2. Auto-fix if needed
# 3. Prompt you to stage the changes
```

### 6. External Tool Documentation

**File**: `tests/test_dependency_enforcement.py::test_external_tools_are_documented`

Scans test files for usage of external CLI tools (via `subprocess.run`, `shutil.which`, etc.) and ensures they're documented in `test_test_dependencies.py`.

If you add code like:

```python
subprocess.run(["some_tool", "--version"])
```

The test will fail if `some_tool` is not in either `REQUIRED_CLI_TOOLS` or `OPTIONAL_CLI_TOOLS`.

## Required Dependencies

### Python Packages (requirements.txt)

**Core**:
- pandas
- numpy
- xlsxwriter
- pydantic>=2
- openpyxl
- PyYAML
- types-PyYAML
- types-requests

**Testing**:
- pytest
- pytest-cov
- pytest-rerunfailures
- hypothesis
- coverage

**Application**:
- streamlit>=1.30
- fastapi>=0.104.0
- httpx>=0.25
- scipy

**Development** (optional):
- black
- ruff
- mypy
- pre-commit

### External CLI Tools

**Required**:
- **Node.js** v20+ (for JavaScript workflow tests)
- **npm** (bundled with Node.js)

**Optional**:
- **uv** (fast Python package installer, for lockfile tests)

## Adding New Dependencies

### For Python Packages

1. Add the package to `requirements.txt`:
   ```
   new-package>=1.0.0
   ```

2. If it's optional (tests can skip gracefully), add to `test_test_dependencies.py`:
   ```python
   OPTIONAL_PYTHON_PACKAGES = [
       ...,
       "new_package",  # Note: use underscore for module name
   ]
   ```

3. Update `requirements.lock` if using uv:
   ```bash
   uv pip compile pyproject.toml -o requirements.lock
   ```

### For External CLI Tools

1. Document in `test_test_dependencies.py`:
   ```python
   REQUIRED_CLI_TOOLS = {
       ...,
       "new_tool": "Description of what it's used for",
   }
   ```

2. Add installation to `.github/workflows/reusable-10-ci-python.yml`:
   ```yaml
   - name: Install new_tool
     run: |
       # Installation commands
   ```

3. Add PATH configuration if needed:
   ```yaml
   echo "$HOME/.local/bin" >> $GITHUB_PATH
   ```

## Test Results

Before implementation:
- **2,056 tests passed, 30 skipped**
- Skipped tests: 19 Node.js tests, 10 keepalive tests, 1 uv test

After implementation:
- **2,102 tests passed, 0 skipped**
- All dependencies installed and validated
- Enforcement tests prevent regression

## Verification Commands

### Check all dependencies are installed:

```bash
./scripts/check_test_dependencies.sh
```

### Run dependency validation tests:

```bash
pytest tests/test_test_dependencies.py -v
```

### Run dependency enforcement tests:

```bash
pytest tests/test_dependency_enforcement.py -v
```

### Run full test suite:

```bash
./scripts/run_tests.sh
```

## Maintenance

### Updating Lockfile

When dependencies change, update the lockfile:

```bash
uv pip compile pyproject.toml -o requirements.lock
```

### Checking for Undeclared Dependencies

The enforcement tests run automatically with every test suite execution. If they fail, it means:

1. A new Python package is imported but not in `requirements.txt`
2. A new CLI tool is used but not documented
3. Requirements.txt has invalid package specifications

Fix by adding the missing dependency to the appropriate configuration file.

## CI Integration

The CI workflow validates dependencies in this order:

1. **Install Python dependencies** from requirements.txt
2. **Install Node.js and npm** via GitHub Actions
3. **Install uv** via official installer
4. **Validate test dependencies** via test suite
5. **Run all tests** (must achieve 0 skipped)

If any dependency is missing, the CI build fails before tests run, making it impossible to merge code with missing dependencies.

## Benefits

1. **Zero skipped tests**: All tests run in all environments
2. **Automatic detection**: Can't add a test without declaring its dependencies
3. **Clear error messages**: Tells developers exactly what to add
4. **CI enforcement**: Prevents incomplete PRs from merging
5. **Documentation**: Self-documenting system via validation tests

## Troubleshooting

### "Module not found" in tests

Run the enforcement test to see what's missing:
```bash
pytest tests/test_dependency_enforcement.py::test_all_test_imports_are_declared -v
```

### "CLI tool not found" in tests

Check which tools are installed:
```bash
./scripts/check_test_dependencies.sh
```

### "Lockfile out of date" error

Regenerate the lockfile:
```bash
uv pip compile pyproject.toml -o requirements.lock
```

### CI failing on dependency installation

Check the CI logs for network timeouts or package availability issues. The setup may take 60-180 seconds.
