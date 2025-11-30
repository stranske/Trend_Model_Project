# Test Dependency Management

> **Canonical documentation**: See [DEPENDENCY_ENFORCEMENT.md](DEPENDENCY_ENFORCEMENT.md) for complete implementation details.

## Quick Reference

### System Status: ENFORCED ✅

The project implements **automatic dependency enforcement** that prevents tests from running with missing dependencies.

**Current Status**: All tests pass with 0 skipped

### Key Features

1. **Automatic Installation**: CI installs all required dependencies (Python packages, Node.js, npm, uv)
2. **Validation Tests**: Test suite validates all dependencies are present before running
3. **Enforcement Tests**: Build fails if new dependencies are used without being declared
4. **Zero Skipped Tests**: All tests must run; skipping due to missing dependencies is not allowed

### Quick Commands

```bash
# Check dependencies locally
./scripts/check_test_dependencies.sh

# Run dependency validation tests
pytest tests/test_test_dependencies.py -v

# Sync missing dependencies
python scripts/sync_test_dependencies.py --fix
uv pip compile pyproject.toml -o requirements.lock
```

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Authoritative dependency declarations |
| `requirements.lock` | Generated pinned dependencies |
| `tests/test_test_dependencies.py` | Validation tests |
| `tests/test_dependency_enforcement.py` | Enforcement tests |
| `scripts/sync_test_dependencies.py` | Auto-sync tool |

### Adding Dependencies

1. Add package to `pyproject.toml` under `[project.optional-dependencies].dev`
2. Run `uv pip compile pyproject.toml -o requirements.lock`
3. Commit both files

For external CLI tools (Node.js, uv, etc.), see [DEPENDENCY_ENFORCEMENT.md](DEPENDENCY_ENFORCEMENT.md).
