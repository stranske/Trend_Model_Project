# Trend Model Project

This repository contains experiments and utilities for analyzing volatility-adjusted trend portfolios. The Jupyter notebooks demonstrate how to load hedge fund data, apply trend-following rules, and export the results.

## Notebooks

- `Vol_Adj_Trend_Analysis1.2.TrEx.ipynb` – an earlier version of the analysis.
- `Vol_Adj_Trend_Analysis1.3.TrEx.ipynb` – the current main notebook showing the full workflow.
- Additional historical notebooks can be found under `notebooks/old` and `Old/`.

## Setup

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file lists common data analysis libraries
   including `pandas`, `numpy`, `matplotlib`, `ipywidgets`, and
   `xlsxwriter`.
2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `Vol_Adj_Trend_Analysis1.3.TrEx.ipynb` and run the cells in order.

## Applying patches

Some updates are provided as patch files. Apply them from the repository root with:

```bash
git apply -p1 <patchfile>
```

Replace `<patchfile>` with the patch you want to apply (for example `codex.patch`).
