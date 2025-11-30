# 📂 `scripts/` — Utility Scripts

> **Purpose:** Automation, development, and CI/CD scripts  
> **Last updated:** November 2025

---

## 🚀 Quick Start Scripts

| Script | Description |
|--------|-------------|
| `setup_env.sh` | Bootstrap virtual environment (60-180s) |
| `run_tests.sh` | Run full test suite with coverage |
| `run_streamlit.sh` | Launch Streamlit web application |
| `generate_demo.py` | Generate demo dataset |

## 🔍 Validation Scripts

| Script | Description |
|--------|-------------|
| `dev_check.sh` | Fast development validation (2-5s) |
| `validate_fast.sh` | Adaptive validation (5-30s) |
| `check_branch.sh` | Comprehensive pre-merge validation (30-120s) |
| `quality_gate.sh` | Quality gate enforcement |
| `quick_check.sh` | Rapid syntax/import check |

## 🔧 CI/CD Scripts

| Script | Description |
|--------|-------------|
| `ci_cosmetic_repair.py` | Auto-fix cosmetic issues |
| `ci_coverage_delta.py` | Calculate coverage changes |
| `ci_history.py` | CI run history tracking |
| `ci_metrics.py` | CI metrics collection |
| `workflow_lint.sh` | Lint GitHub workflow files |
| `workflow_smoke_tests.py` | Workflow smoke tests |

## 📊 Analysis & Reporting

| Script | Description |
|--------|-------------|
| `run_multi_demo.py` | Multi-period demo runner |
| `walk_forward.py` | Walk-forward analysis |
| `benchmark_performance.py` | Performance benchmarking |
| `compare_perf.py` | Performance comparison |
| `generate_residual_report.py` | Residual analysis reports |

## 🤖 Automation

| Script | Description |
|--------|-------------|
| `codex_git_bootstrap.sh` | Codex agent git setup |
| `keepalive-runner.js` | Keepalive workflow runner |
| `open_pr_from_issue.sh` | Create PR from issue |
| `git_hooks.sh` | Install git hooks |

## 🛠️ Development Tools

| Script | Description |
|--------|-------------|
| `fix_common_issues.sh` | Auto-fix common problems |
| `mypy_autofix.py` | Auto-fix type errors |
| `prune_allowlist.py` | Prune lint allowlists |
| `sync_tool_versions.py` | Sync tool versions |

---

## 📋 Common Workflows

### Development Cycle
```bash
./scripts/dev_check.sh --changed --fix  # Quick validation
./scripts/validate_fast.sh --fix        # Before commit
./scripts/check_branch.sh --fast --fix  # Before merge
```

### Demo Pipeline
```bash
./scripts/setup_env.sh
python scripts/generate_demo.py
python scripts/run_multi_demo.py
```

---

*See `.github/copilot-instructions.md` for timing expectations.*
