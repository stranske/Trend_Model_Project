# 📂 `data/` — Reference Datasets

> **Purpose:** Demo datasets and reference data for tests and examples  
> **Last updated:** November 2025

---

## 📊 Primary Datasets

| File | Size | Description |
|------|------|-------------|
| `Trend Universe Data.csv` | 136 KB | Monthly total returns for Trend universe funds |
| `Trend Universe Membership.csv` | 4 KB | Fund membership effective dates |
| `hedge_fund_returns_with_indexes.csv` | 140 KB | Hedge fund returns with benchmark indices |

## 📁 Subdirectories

### `raw/`
Raw input data organized by type:

#### `raw/managers/`
- `sample_manager.csv` — Minimal manager fixture for tests

#### `raw/indices/`
- `sample_index.csv` — Sample benchmark index data

---

## 📋 Schema Reference

### Trend Universe Data
- **Columns:** `Date` + fund/index return series (decimal percentages)
- **Frequency:** Monthly
- **Pair with:** Membership ledger for effective windows

### Trend Universe Membership
- **Columns:** `fund`, `effective_date`, `end_date`
- **Purpose:** Maps each fund to its active date range

### Hedge Fund Returns
- **Columns:** `Date`, `Risk-Free Rate`, fund returns
- **Frequency:** Monthly
- **Use:** Long backtests and rolling-hold configs

---

## ⚠️ Important Notes

1. **Synthetic Data:** All datasets are synthetic or derived from public benchmarks
2. **Demo Only:** Not suitable for production trading decisions
3. **Provenance:** See `README_DATA.md` for full details

---

*See `docs/data/Trend_Universe_Data.md` for stewardship notes.*
