# Dependency Synchronization Process

## Overview

Test dependencies are automatically synchronized across the repository to ensure:
1. All test file imports are declared in `requirements.txt`
2. No phantom dependencies (tests passing locally but failing in CI)
3. Consistent dependency management across all environments

## Automated Synchronization

### CI Check (Gate Workflow)

**Location**: `.github/workflows/reusable-10-ci-python.yml`

The CI workflow includes a dependency verification step that:
1. Scans all test files for imports
2. Compares discovered imports against declared dependencies in `requirements.txt`
3. **Fails the build** if undeclared dependencies are found
4. Provides instructions to run `sync_test_dependencies.py --fix`

**When it runs**: On every PR commit as part of the Gate workflow

### Autofix Workflow

**Location**: `.github/workflows/autofix.yml`

The autofix workflow automatically:
1. Runs `sync_test_dependencies.py --fix` on PRs with Python files
2. Adds missing dependencies to `requirements.txt`
3. Commits and pushes changes **directly to the PR branch**
4. Respects `autofix:pause` label to skip automatic fixes
5. Limits to 2 attempts per commit to prevent infinite loops

**When it runs**: 
- Triggered by CI Autofix Loop workflow on PR events
- Only for same-repository PRs (not forks)
- Only when Python files (`.py`, `.pyi`) are changed

## Manual Synchronization

### Check for Missing Dependencies

```bash
python scripts/sync_test_dependencies.py --verify
```

**Output**:
- ✅ Success: All dependencies declared
- ❌ Failure: Lists undeclared dependencies

### Fix Missing Dependencies

```bash
python scripts/sync_test_dependencies.py --fix
```

**Actions**:
- Adds missing packages to `requirements.txt` (alphabetically under `# Testing dependencies`)
- Preserves existing formatting and comments
- Reports what was added

### Dry Run (Preview Changes)

```bash
python scripts/sync_test_dependencies.py
```

Shows what would be changed without modifying files.

## How It Works

### 1. Import Detection

The sync script (`scripts/sync_test_dependencies.py`):
- Scans all `*.py` files in `tests/` directory
- Parses imports using Python AST (Abstract Syntax Tree)
- Extracts top-level module names from:
  - `import module`
  - `from module import ...`

### 2. Filtering

Discovered imports are filtered against:

**Standard Library Modules** (`STDLIB_MODULES`)
- Built-in Python modules (e.g., `os`, `sys`, `json`)
- Don't need to be installed

**Test Framework Modules** (`TEST_FRAMEWORK_MODULES`)
- `pytest`, `hypothesis`, `_pytest`, `pluggy`
- Core test infrastructure

**Project Modules** (`PROJECT_MODULES`)
- Local source code modules (e.g., `trend_analysis`)
- Local scripts (e.g., `health_summarize`, `gate_summary`)
- Not PyPI packages

**Module-to-Package Mappings** (`MODULE_TO_PACKAGE`)
- `yaml` → `PyYAML`
- `PIL` → `Pillow`
- `sklearn` → `scikit-learn`
- etc.

### 3. Dependency Resolution

After filtering:
1. Remaining imports are considered external dependencies
2. Mapped to package names via `MODULE_TO_PACKAGE`
3. Compared against `requirements.txt`
4. Missing packages are added

## Adding New Project Scripts

If you add a new local script that tests import:

1. Add the module name to `PROJECT_MODULES` in `sync_test_dependencies.py`
2. Example:
   ```python
   PROJECT_MODULES = {
       ...
       "my_new_script",  # .github/scripts/my_new_script.py
   }
   ```

**Recent Example**: `health_summarize` was added to prevent it from being treated as a PyPI package.

## Configuration Files

### scripts/sync_test_dependencies.py

**Purpose**: Main synchronization logic

**Key Sections**:
- `STDLIB_MODULES`: Python standard library modules to ignore
- `TEST_FRAMEWORK_MODULES`: Test infrastructure to ignore
- `PROJECT_MODULES`: Local modules/scripts to ignore
- `MODULE_TO_PACKAGE`: Import name → package name mappings

**Functions**:
- `extract_imports_from_file()`: Parse imports from a file
- `get_all_test_imports()`: Scan all test files
- `get_declared_dependencies()`: Read requirements.txt
- `get_undeclared_dependencies()`: Find missing dependencies
- `add_dependencies_to_requirements()`: Add missing deps

### requirements.txt

**Structure**:
```txt
# Core dependencies
pandas
numpy

# Testing dependencies
pytest>=8.0
pytest-cov
<-- new test deps added here -->

# Documentation
sphinx
```

Dependencies are added alphabetically under `# Testing dependencies` section.

## Troubleshooting

### False Positive: Local Script Detected as Missing Package

**Symptom**: CI fails with "undeclared dependency: my_script"

**Cause**: Test file imports a local script, but `sync_test_dependencies.py` doesn't know it's local

**Fix**: Add the module name to `PROJECT_MODULES` in `sync_test_dependencies.py`

### Autofix Not Running

**Possible Causes**:
1. **PR is from a fork**: Autofix only runs on same-repository PRs
2. **PR has `autofix:pause` label**: Remove the label
3. **No Python files changed**: Autofix only runs when `.py` or `.pyi` files are modified
4. **Max attempts reached**: Autofix limits to 2 attempts per commit
5. **PR is draft**: Autofix skips draft PRs

### Autofix Creates Side Branches

**Status**: This was reported but investigation shows autofix pushes to PR branches correctly.

**Evidence**: Line 569 of `autofix.yml` pushes to `HEAD:${{ steps.context.outputs.head_ref }}` where `head_ref` is `pr.head.ref` (the PR branch name).

**If still occurring**: Check recent autofix run logs for actual branch push targets.

## Best Practices

### For Contributors

1. **Run locally before committing**:
   ```bash
   python scripts/sync_test_dependencies.py --verify
   ```

2. **Let autofix handle it**: If you forget, autofix will add missing deps automatically

3. **Don't manually add test deps**: Let the script maintain consistency

### For Maintainers

1. **Keep exclusion lists updated**: Add new local scripts to `PROJECT_MODULES`

2. **Monitor autofix runs**: Check if it's actually fixing issues or hitting limits

3. **Update mappings**: Add new module-to-package mappings as needed

4. **Review autofix commits**: Ensure added dependencies are correct

## Related Documentation

- [CI/CD Documentation](../ci/)
- [Testing Guidelines](TESTING_SUMMARY.md)
- [Autofix Workflow](../.github/workflows/autofix.yml)
- [Dependency Check](../.github/workflows/reusable-10-ci-python.yml)

## Change Log

### 2025-10-30
- Added `health_summarize` to `PROJECT_MODULES` to prevent false dependency detection
- Documented autofix behavior (pushes to PR branches, not side branches)
- Created this documentation

### Earlier
- Implemented `sync_test_dependencies.py` script
- Added CI verification step
- Integrated with autofix workflow
