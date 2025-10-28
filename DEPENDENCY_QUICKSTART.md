# Dependency Management - Quick Reference

## 🚀 Quick Start

### I just added an import to a test file. What do I do?

```bash
# 1. Check if it's declared
python scripts/sync_test_dependencies.py

# 2. If missing, auto-fix
python scripts/sync_test_dependencies.py --fix

# 3. Update lockfile
uv pip compile pyproject.toml -o requirements.lock

# 4. Commit
git add requirements.txt requirements.lock
git commit -m "Add test dependency for X"
```

## 🎯 Common Commands

### Check Status
```bash
# See what dependencies would be added (dry run)
python scripts/sync_test_dependencies.py
```

**Output if everything is good:**
```
✅ All test dependencies are declared in requirements.txt
```

**Output if missing dependencies:**
```
❌ Found undeclared imports in tests:
   - requests (used in 5 test files)
   - hypothesis (used in 3 test files)

Run with --fix to automatically add these to requirements.txt
```

### Fix Missing Dependencies
```bash
# Automatically add missing packages
python scripts/sync_test_dependencies.py --fix

# Then update lockfile
uv pip compile pyproject.toml -o requirements.lock
```

### Run Validation Tests
```bash
# Check all dependency tests pass
python -m pytest tests/test_test_dependencies.py tests/test_dependency_enforcement.py -v
```

## 🔧 Installation

### Optional: Pre-commit Hook
Automatically check dependencies before each commit:

```bash
cp scripts/pre-commit-check-deps.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Now every `git commit` will:
1. Check for undeclared dependencies
2. Auto-fix if issues found
3. Prompt you to stage the changes

## 🚨 CI Failed - What Now?

### Error: "test_all_test_imports_are_declared FAILED"

This means your PR adds imports that aren't declared in `requirements.txt`.

**Fix locally:**
```bash
# Auto-fix
python scripts/sync_test_dependencies.py --fix
uv pip compile pyproject.toml -o requirements.lock

# Commit and push
git add requirements.txt requirements.lock
git commit -m "Declare test dependencies"
git push
```

**Or check CI job summary** - it shows exactly what needs to be added.

## 📋 What Gets Checked

### ✅ Included in Enforcement
- All imports in `tests/**/*.py` files
- Third-party packages (numpy, pandas, requests, etc.)

### ⏭️ Excluded from Enforcement  
- Python standard library (json, os, sys, etc.)
- Test frameworks (pytest, unittest)
- Project modules (trend_analysis, trend_portfolio_app)

## 🎓 Module Name vs Package Name

Some packages have different module and package names:

| Import Statement | Package Name in requirements.txt |
|-----------------|----------------------------------|
| `import yaml` | `PyYAML` |
| `from PIL import Image` | `Pillow` |
| `import cv2` | `opencv-python` |
| `import pre_commit` | `pre-commit` |

The sync script handles these automatically.

## 📚 Full Documentation

- **Complete Guide**: `docs/DEPENDENCY_SYSTEM_COMPLETE.md`
- **Workflow Details**: `docs/DEPENDENCY_WORKFLOW.md`
- **Technical Deep Dive**: `docs/DEPENDENCY_ENFORCEMENT.md`

## ⚡ TL;DR

1. Added import to test? Run `python scripts/sync_test_dependencies.py --fix`
2. Update lockfile: `uv pip compile pyproject.toml -o requirements.lock`
3. Commit both files
4. Done! ✅

---

**Questions?** Check `docs/DEPENDENCY_WORKFLOW.md` for step-by-step instructions.
