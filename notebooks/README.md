# Notebooks

## Maintained files
- `Vol_Adj_Trend_Analysis1.5.TrEx.ipynb`: current reference notebook for running and documenting the volatility-adjusted trend pipeline. Keep it in sync with the latest production configuration and demos referenced in `docs/quickstart.md`.

## Maintenance expectations
- **Update cadence:** refresh outputs and narrative at least once per release cycle or after material pipeline changes (e.g., new metrics, selection modes, or data-quality filters).
- **Data sources:** default inputs are the Trend Universe data files (CSV/Excel) under the project root. If alternative datasets are used, document paths and preprocessing steps in the notebook header.
- **Review checklist:** ensure widgets and configuration examples match `config/` defaults, rerun all cells before committing, and clear temporary paths or credentials.

## Archives
Older exploratory and superseded notebooks have been moved to `archives/notebooks/2025`. Start new explorations in this folder and archive them once stabilized or replaced.
