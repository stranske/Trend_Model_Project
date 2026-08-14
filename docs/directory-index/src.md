# 📂 `src/` — Source Code Directory

> **Purpose:** Main application source code  
> **Last updated:** November 2025

---

## 📦 Packages

| Package | Description |
|---------|-------------|
| `trend_analysis/` | Core trend analysis engine and pipeline |
| `trend/` | Trend model implementation and signal generation |
| `trend_portfolio_app/` | Portfolio application components |
| `backtest/` | Backtesting framework |
| `data/` | Data loading and validation |
| `health_summarize/` | Health check summarization |
| `utils/` | Shared utilities |

## 📄 Root Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization |
| `cli.py` | Command-line interface entry point |

---

## 🔗 Key Subpackages

### `trend_analysis/`
The primary analysis package containing:
- Pipeline orchestration
- Metrics computation
- Configuration management
- Multi-period analysis engine
- Export functionality

### `trend_portfolio_app/`
Streamlit web application components for interactive portfolio analysis.

### `backtest/`
Walk-forward and backtesting utilities for strategy validation.

---

*See `docs/architecture.md` for detailed module relationships.*
