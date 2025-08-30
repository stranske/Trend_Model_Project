# Release Process Documentation

This document describes the automated release process for the Trend Analysis package to PyPI.

## Overview

The repository includes an automated CI/CD pipeline that:
- Builds distribution packages (wheels and source distributions) 
- Tests package installation
- Generates changelogs from git history
- Publishes releases to PyPI
- Creates GitHub releases with changelog

## Workflow Triggers

The release workflow (`.github/workflows/release.yml`) can be triggered in two ways:

### 1. Automatic Release on Version Tags

Push a semantic version tag to trigger automatic release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Supported tag formats:
- `v1.0.0` - Major release
- `v0.1.0` - Minor release  
- `v0.0.1` - Patch release
- `v1.0.0-alpha.1` - Pre-release (marked as prerelease on GitHub)

### 2. Manual Release via GitHub Actions

Use the GitHub Actions interface for controlled releases:

1. Go to **Actions** tab in the GitHub repository
2. Select the **Release** workflow
3. Click **Run workflow**
4. Specify:
   - **Tag**: Version tag (e.g., `v0.1.0`)
   - **Dry run**: Check to test on TestPyPI only

## Release Steps

The workflow performs the following steps:

### Build Stage
1. **Checkout code** with full git history for changelog generation
2. **Set up Python** 3.11 environment
3. **Install build tools** (`build`, `twine`)
4. **Extract version** from git tag (removes `v` prefix)
5. **Update pyproject.toml** with extracted version
6. **Build packages** using `python -m build`
7. **Validate packages** using `twine check`
8. **Upload build artifacts** for testing stages

### Test Installation Stage
1. **Download build artifacts**
2. **Test wheel installation** and basic import
3. **Test source distribution** installation and import

### TestPyPI Stage (Dry Run Only)
1. **Publish to TestPyPI** for testing
2. Only runs when `dry_run` input is true

### Release Stage (Production)
1. **Generate changelog** using `git-cliff`
2. **Create GitHub release** with changelog and installation instructions
3. **Upload release assets** (wheel and source distribution files)
4. **Publish to PyPI** using trusted publishing

## Prerequisites

### Repository Secrets

The following secrets must be configured in the GitHub repository:

- `PYPI_API_TOKEN` - PyPI API token for publishing to production PyPI
- `TEST_PYPI_API_TOKEN` - TestPyPI API token for dry run testing

### PyPI Trusted Publishing (Recommended)

For enhanced security, configure PyPI trusted publishing:

1. Create publisher on PyPI:
   - Repository: `stranske/Trend_Model_Project`
   - Workflow: `release.yml`
   - Environment: `pypi`

2. Create publisher on TestPyPI (for testing):
   - Repository: `stranske/Trend_Model_Project` 
   - Workflow: `release.yml`
   - Environment: `test-pypi`

## Changelog Generation

The workflow uses `git-cliff` to automatically generate changelogs based on:

- **Conventional Commits** - Commits following the format: `type(scope): description`
- **Git tag ranges** - Changes between current and previous version tags
- **Configuration** - Changelog format defined in `cliff.toml`

### Supported Commit Types

- `feat:` - New features (🚀 Features section)
- `fix:` - Bug fixes (🐛 Bug Fixes section)
- `docs:` - Documentation changes (📚 Documentation section)
- `refactor:` - Code refactoring (🚜 Refactor section)
- `perf:` - Performance improvements (⚡ Performance section)
- `test:` - Testing changes (🧪 Testing section)
- `chore:` - Maintenance tasks (⚙️ Miscellaneous Tasks section)

## Testing Releases

### Dry Run to TestPyPI

Test the release process without affecting production:

```bash
# Trigger via GitHub UI with dry_run=true
# Or manually via GitHub CLI:
gh workflow run release.yml -f tag=v0.1.0-test -f dry_run=true
```

### Local Testing

Test package building locally:

```bash
# Install build tools
pip install build twine

# Build packages
python -m build

# Check packages
twine check dist/*

# Test local installation
pip install dist/*.whl
python -c "import trend_analysis; print(trend_analysis.__version__)"
```

## Package Information

- **Name**: `trend-analysis`
- **PyPI URL**: https://pypi.org/project/trend-analysis/
- **TestPyPI URL**: https://test.pypi.org/project/trend-analysis/

## CLI Commands

The package provides CLI commands after installation:

```bash
# Single-period analysis
trend-analysis -c config.yml

# Multi-period analysis  
trend-multi-analysis -c config.yml
```

## Troubleshooting

### Build Failures
- Check `pyproject.toml` syntax and dependencies
- Verify all package files are included in git
- Ensure version is valid semantic version format

### Publication Failures
- Verify PyPI API tokens are valid and have correct permissions
- Check that package version doesn't already exist on PyPI
- Ensure trusted publishing is configured correctly

### Changelog Issues
- Use conventional commit messages for automatic categorization
- Check `cliff.toml` configuration for custom formatting
- Ensure git tags are pushed before triggering release

## Version Management

The package version is managed in `pyproject.toml` and automatically updated during release:

1. Developer creates git tag with desired version
2. Workflow extracts version from tag (removing `v` prefix)
3. `pyproject.toml` is updated with extracted version
4. Package is built with updated version
5. Version is accessible via `trend_analysis.__version__`

## Manual Release Process

If automated release fails, manual steps:

```bash
# 1. Update version in pyproject.toml
sed -i 's/version = ".*"/version = "1.0.0"/' pyproject.toml

# 2. Build packages
python -m build

# 3. Check packages  
twine check dist/*

# 4. Upload to PyPI
twine upload dist/*

# 5. Create git tag and GitHub release
git tag v1.0.0
git push origin v1.0.0
```