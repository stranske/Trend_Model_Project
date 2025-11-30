# 📂 `tests/` — Test Suite

> **Purpose:** Comprehensive unit and integration tests  
> **Last updated:** November 2025  
> **Test count:** 400+ tests | **Coverage target:** 70%+

---

## 📁 Structure

| Directory | Description |
|-----------|-------------|
| `app/` | Application-level tests |
| `backtesting/` | Backtest engine tests |
| `data/` | Data loading/validation tests |
| `fixtures/` | Test fixtures and sample data |
| `github_scripts/` | GitHub workflow script tests |
| `golden/` | Golden master comparison files |
| `proxy/` | Proxy server tests |
| `scripts/` | Script tests |
| `smoke/` | Smoke tests for quick validation |
| `soft_coverage/` | Soft coverage tracking |
| `tools/` | Tool tests |
| `trend_analysis/` | Core analysis tests |
| `unit/` | Pure unit tests |

---

## 🧪 Test Categories

### Core Analysis
- `test_pipeline*.py` — Pipeline orchestration
- `test_metrics*.py` — Financial metrics
- `test_config*.py` — Configuration loading
- `test_export*.py` — Export functionality

### Multi-Period Engine
- `test_multi_period_engine*.py` — Rolling analysis engine
- `test_multi_period_export.py` — Period export tests
- `test_multi_period_selection.py` — Manager selection

### Data & Validation
- `test_data*.py` — Data loading
- `test_validators*.py` — Input validation
- `test_market_data*.py` — Market data handling

### UI & Application
- `test_streamlit*.py` — Streamlit app tests
- `test_gui*.py` — GUI component tests
- `test_cli*.py` — CLI tests

### Workflows & CI
- `test_workflow*.py` — Workflow tests
- `test_autofix*.py` — Autofix pipeline tests
- `test_keepalive*.py` — Keepalive tests

---

## 🚀 Running Tests

```bash
# Full suite with coverage
./scripts/run_tests.sh

# Quick run
pytest -q

# Specific category
pytest tests/test_pipeline*.py

# With coverage report
pytest --cov=trend_analysis --cov-branch
```

---

## 📋 Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures and configuration |
| `quarantine.yml` | Quarantined flaky tests |
| `sitecustomize.py` | Test environment customization |
| `score_frame_golden.csv` | Golden master for score frame |

---

*See `pytest.ini` for test configuration.*
