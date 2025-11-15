# Trend Analysis Project - GitHub Copilot Instructions

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Project Overview

Trend Analysis Project is a Python financial analysis application that provides volatility-adjusted trend portfolio analysis. It includes a command-line interface, Jupyter notebooks, Streamlit web application, and comprehensive testing infrastructure with multi-tier validation.

## Working Effectively

### Collaboration Rhythm (User Preference)
- Plough through the active todo list continuously until a genuine blocker occurs or a decision point materially affects future work.
- Do **not** pause or conclude a session without first asking the specific question needed to proceed or stating what information is required to continue.
- Surface forks in the plan clearly, outlining the options and the inputs needed to choose between them.
- Only yield control after receiving the missing info, resolving the blocker, or finishing every item on the todo list.

### Bootstrap Environment
**NEVER CANCEL: Environment setup takes 60-180 seconds. Set timeout to 300+ seconds.**
```bash
./scripts/setup_env.sh
```
This creates `.venv/` virtual environment and installs all dependencies from `requirements.lock` via `uv pip sync`. The script is safe to source or execute.

**Network Timeout Issues**: The setup may fail due to PyPI connectivity timeouts. If this happens:
- Environment setup: fails with `ReadTimeoutError` from PyPI
- Alternative: Use system Python packages or pre-configured environment if available
- The timeout is a network infrastructure limitation, not a code issue

### Build and Test
**NEVER CANCEL: Test suite takes 15-25 seconds. Set timeout to 60+ seconds.**
```bash
# Run full test suite with coverage (248 tests, requires 70%+ coverage)
./scripts/run_tests.sh

# Alternative: Run tests directly with PYTHONPATH
source .venv/bin/activate
PYTHONPATH="./src" pytest --cov trend_analysis --cov-branch
```

### Generate Demo Dataset
```bash
# Creates demo/demo_returns.csv and demo/demo_returns.xlsx (5-10 seconds)
python scripts/generate_demo.py
```

## Application Execution

### Command-Line Analysis
```bash
# Activate environment first
source .venv/bin/activate

# Run analysis with config file (completes in <1 second)
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml

# Run with specific config file via environment variable (e.g. defaults)
TREND_CFG=config/defaults.yml PYTHONPATH="./src" python -m trend_analysis.run_analysis

# Run with custom config file via environment variable
TREND_CFG=custom.yml PYTHONPATH="./src" python -m trend_analysis.run_analysis
```

### Streamlit Web Application
```bash
# Start interactive web app (starts in 3-5 seconds)
./scripts/run_streamlit.sh

# For headless mode
./scripts/run_streamlit.sh --server.headless true
```
Access at http://localhost:8501 after startup.

### Jupyter Notebooks
```bash
# After environment setup
source .venv/bin/activate
jupyter lab
# or
jupyter notebook
```
Main notebook: `Vol_Adj_Trend_Analysis1.5.TrEx.ipynb`

## Validation Workflow (Critical)

**ALWAYS use the multi-tier validation system based on development stage:**

### Tier 1: Ultra-Fast Development Validation (2-5 seconds)
```bash
# Perfect for active development - only checks changed files
./scripts/dev_check.sh --changed --fix
```

### Tier 2: Intelligent Adaptive Validation (5-30 seconds)
```bash
# Auto-selects validation strategy based on changes
./scripts/validate_fast.sh --fix
```
- **Incremental** (1-3 files): 5-15s
- **Comprehensive** (config/src changes): 15-30s  
- **Full** (major changes): 30-60s

### Tier 3: Comprehensive Pre-Merge Validation (30-120 seconds)
**NEVER CANCEL: Comprehensive validation takes 30-120 seconds. Set timeout to 180+ seconds.**
```bash
# Complete validation before merging
./scripts/check_branch.sh --fast --fix

# Full validation for CI/production  
./scripts/check_branch.sh
```

**What comprehensive validation checks:**
- ✅ Code formatting (Black)
- ✅ Linting (Flake8)
- ✅ Type checking (MyPy) 
- ✅ Package installation
- ✅ Import validation
- ✅ Unit tests
- ✅ Test coverage (70% minimum)
- ✅ Git status and branch info

## Validation Requirements

**ALWAYS run validation before committing changes:**
```bash
# Quick validation during development (2-5s)
./scripts/dev_check.sh --changed --fix

# Before committing (5-15s)
./scripts/validate_fast.sh --fix

# Before merging (30-90s)
./scripts/check_branch.sh --fast --fix
```

**Auto-fix capabilities:**
```bash
# Fix formatting and common issues automatically
./scripts/validate_fast.sh --fix
./scripts/fix_common_issues.sh
```

## Manual Validation Checklist

**After making changes, ALWAYS run these validation steps:**

### Quick Development Check (2-5 seconds)
```bash
./scripts/dev_check.sh --changed --fix
```
Verifies: Syntax, imports, changed files only

### Comprehensive Validation (30-120 seconds)  
**NEVER CANCEL: Set timeout to 180+ seconds**
```bash
./scripts/check_branch.sh --fast --fix
```
Verifies: Formatting, linting, type checking, imports, git status

### User Scenario Testing
**ALWAYS test at least one complete scenario after changes:**

**Scenario 1: Basic CLI Analysis** (requires dependencies)
```bash
# 1. Generate demo data
python scripts/generate_demo.py

# 2. Run analysis
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml

# 3. Verify output shows fund metrics (CAGR, Sharpe ratio, etc.)
```

**Scenario 2: Repository Structure Validation** (always works)
```bash
# 1. Check key directories exist
ls -la src/trend_analysis/ tests/ scripts/ config/

# 2. Verify scripts are executable  
./scripts/validate_fast.sh
./scripts/check_branch.sh --fast

# 3. Check git status
git status
```

### Scenario 3: CLI Trend Analysis
```bash
source .venv/bin/activate
python scripts/generate_demo.py
# Run the main analysis command (see "Common Commands" section above)
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml
# Verify output shows fund analysis with metrics like CAGR, Sharpe ratio, etc.
```

### Scenario 2: Streamlit Interactive Analysis
```bash
./scripts/run_streamlit.sh --server.headless true &
# Wait 5 seconds for startup
curl -f http://localhost:8501 || echo "Streamlit not responding"
./scripts/run_streamlit.sh --server.headless true & STREAMLIT_PID=$!
# Wait 5 seconds for startup
curl -f http://localhost:8501 || echo "Streamlit not responding"
# Kill background process: kill $STREAMLIT_PID
```

### Scenario 3: Complete Development Workflow
```bash
# 1. Make code changes
# 2. Fast validation (must complete in <15s)
./scripts/dev_check.sh --changed --fix
# 3. Test specific functionality
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml
# 4. Comprehensive validation before commit
./scripts/validate_fast.sh --fix
# 5. Full test suite
./scripts/run_tests.sh
```

## Key Repository Locations

### Core Package Structure
- `src/trend_analysis/` - Main analysis package
  - `src/trend_analysis/config.py` - Configuration loading
  - `src/trend_analysis/pipeline.py` - Analysis pipeline
  - `src/trend_analysis/metrics.py` - Financial metrics
  - `src/trend_analysis/export.py` - Data export functionality
  - `src/trend_analysis/gui/` - Interactive GUI components
  - `src/trend_analysis/multi_period/` - Multi-period analysis engine
- `src/trend_portfolio_app/` - Streamlit web application
- `tests/` - Unit tests (248 tests, 71% coverage)
- `scripts/` - Utility and validation scripts
- `config/` - YAML configuration files
- `demo/` - Generated demo datasets

### Configuration Files
- `requirements.lock` - Generated pinned dependencies  
- `pyproject.toml` - Build system and tool configuration
- `config/defaults.yml` - Default analysis configuration
- `config/demo.yml` - Demo analysis configuration
- `.flake8` - Linting configuration
- `.pre-commit-config.yaml` - Pre-commit hooks

### Important Scripts
- `scripts/setup_env.sh` - Environment setup (60-90s)
- `scripts/run_tests.sh` - Test execution (15-25s)
- `scripts/generate_demo.py` - Demo data generation (5-10s)
- `scripts/run_streamlit.sh` - Streamlit app launcher
- `scripts/dev_check.sh` - Fast development validation (2-5s)
- `scripts/validate_fast.sh` - Adaptive validation (5-30s)  
- `scripts/check_branch.sh` - Comprehensive validation (30-120s)

## Critical Timing Expectations

**NEVER CANCEL these operations - use specified timeout values:**

| Operation | Typical Time | Timeout Setting | Command |
|-----------|--------------|-----------------|---------|
| Environment Setup | 60-180s (may fail on timeouts) | 300s | `./scripts/setup_env.sh` |
| Test Suite | 15-25s | 60s | `./scripts/run_tests.sh` |
| Demo Generation | 5-10s | 30s | `python scripts/generate_demo.py` |
| CLI Analysis | <1s | 30s | `PYTHONPATH="./src" python -m trend_analysis.run_analysis` |
| Fast Validation | 2-5s | 15s | `./scripts/dev_check.sh --changed --fix` |
| Adaptive Validation | 5-30s | 60s | `./scripts/validate_fast.sh --fix` |
| Comprehensive Validation | 30-120s | 180s | `./scripts/check_branch.sh --fast` |
| Streamlit Startup | 3-5s | 30s | `./scripts/run_streamlit.sh` |

## Development Workflow Integration

### Git Hooks (Optional)
```bash
# Install automatic validation hooks
./scripts/git_hooks.sh install
```
- Pre-commit: Fast validation (5-15s)
- Pre-push: Comprehensive validation (30-90s)

### IDE Integration
For VS Code, add to `tasks.json`:
```json
{
    "label": "Quick Validation", 
    "type": "shell",
    "command": "./scripts/dev_check.sh --fix",
    "group": "build"
}
```

## Common Tasks Reference

### Repository Structure
```
/home/runner/work/Trend_Model_Project/Trend_Model_Project/
├── src/trend_analysis/           # Core package
├── src/trend_portfolio_app/      # Streamlit app  
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── config/                       # Configuration files
├── demo/                         # Generated datasets
├── docs/                         # Documentation
├── requirements.lock             # Generated dependency lock
├── pyproject.toml               # Build configuration
└── .github/                     # GitHub workflows
```

### Frequently Used Commands
```bash
# Complete setup from scratch (300s timeout, may fail on network issues)
./scripts/setup_env.sh

# Development iteration cycle (total <20s, works with existing packages)
./scripts/dev_check.sh --changed --fix
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml
./scripts/validate_fast.sh --fix

# Pre-commit validation (60s timeout, needs existing packages)  
./scripts/validate_fast.sh --fix
./scripts/run_tests.sh

# Pre-merge validation (180s timeout, needs existing packages)
./scripts/check_branch.sh --fast --fix
```

## Troubleshooting

### Network Connectivity Issues
**CRITICAL**: Environment setup may fail due to PyPI timeouts:
```
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.
```

**Workarounds**:
- Wait and retry: `./scripts/setup_env.sh`  
- Use system packages: `python3 -m pip install --user pandas numpy PyYAML xlsxwriter pytest`
- Manual install: `pip install pandas numpy PyYAML xlsxwriter openpyxl pydantic pytest pytest-cov streamlit`
- Skip pip install failures but document what doesn't work
- The core analysis requires: pandas, numpy, PyYAML, xlsxwriter

**Status when dependencies missing**:
- ✗ Package import: `ModuleNotFoundError: No module named 'numpy'`
- ✗ CLI analysis: Will fail on import  
- ✗ Test suite: Will fail on import
- ✓ Validation scripts: Work without dependencies (but can't validate linting without black/flake8)
- ✓ Repository structure: Can be explored normally

### Common Issues
- **Module not found**: Use `PYTHONPATH="./src"` prefix or `pip install -e .` if pip works
- **Permission denied on scripts**: Run `chmod +x scripts/*.sh`
- **Test failures**: Check with `./scripts/check_branch.sh --verbose`
- **Validation failures**: Use `--fix` flags to auto-correct formatting issues
- **PyPI timeouts**: Environment setup may take multiple attempts due to network limitations

### Environment Variables
```bash
# Skip certain validation checks
export SKIP_COVERAGE=1
export SKIP_MYPY=1

# Adjust validation thresholds  
export COVERAGE_THRESHOLD=65
export MAX_LINE_LENGTH=88
```

## Performance Optimization Notes

The validation ecosystem provides **8-16x faster development velocity** through intelligent tiering:
- **Active Development**: 2-5s validation vs 60-120s previously
- **Code Changes**: 5-30s validation vs 60-120s previously  
- **Pre-Commit**: 15-30s validation vs 60-120s previously

Use the appropriate tier for your development stage to maintain rapid iteration while ensuring code quality.