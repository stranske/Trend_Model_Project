# Dependency Management - Quick Reference

The project now relies on **pyproject.toml** as the single source of truth for
runtime, development, and application dependencies. All reproducible installs
flow from that file and the generated `requirements.lock`.

## 🚀 Quick Start

### Added a new third-party import in tests?

```bash
# 1. Detect missing packages
python scripts/sync_test_dependencies.py

# 2. Auto-declare any missing packages in the dev extra
python scripts/sync_test_dependencies.py --fix

# 3. Regenerate the lock file from pyproject.toml
make lock

# 4. Commit the updated metadata
git add pyproject.toml requirements.lock
```

The sync script updates `[project.optional-dependencies].dev` so that all
CI/dev tooling stays in sync. The lock file is generated with
`uv pip compile` under the hood via the `make lock` shortcut.

## 🎯 Common Commands

### Check Status
```bash
python scripts/sync_test_dependencies.py            # dry-run check
python scripts/sync_test_dependencies.py --verify   # exit 1 on drift (CI mode)
```

### Regenerate the lock file
```bash
make lock
```

### Validate the tooling pins
```bash
python scripts/sync_tool_versions.py --check   # ensure pyproject pins match the pin file
python scripts/sync_tool_versions.py --apply   # rewrite pyproject.toml when pins drift
```

## 🔧 Installation Paths

All installs consume `pyproject.toml` plus the generated lock file:

* **Local development**
  ```bash
  uv pip sync requirements.lock
  pip install --no-deps -e .[dev]
  ```

* **CI**
  ```bash
  uv pip sync requirements.lock
  pip install --no-deps -e .[dev]
  ```

* **Docker** – the Dockerfile installs build tools, runs `uv pip sync
  requirements.lock`, and finishes with `pip install --no-deps -e .[app]`.

`pip install -e .` on a fresh virtualenv should not resolve any new versions
when the lock has already been synced.

## 📚 Documentation

* **Detailed Workflow:** `docs/DEPENDENCY_WORKFLOW.md`
* **Lock File Contract:** `docs/DEPENDENCY_SYSTEM_COMPLETE.md`
* **Troubleshooting:** `docs/DEPENDENCY_ENFORCEMENT.md`

## ⚡ TL;DR

1. Declare dependencies in `pyproject.toml` (sync script helps for tests).
2. Run `make lock` to regenerate `requirements.lock`.
3. Commit both files so CI, Docker, and local dev stay in sync.
