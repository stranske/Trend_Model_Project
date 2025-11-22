# Inventory and archiving log (2025-11-22)

## Static reference sweep (cutoff: 6 months)
- Method: scanned `git log --since=2025-05-22 --name-only` to find files with no commits in the window and used `rg` to check for inbound references by filename.
- Findings:
  - No files exceeded the 6-month age threshold (repository activity is newer than the cutoff).
  - Unreferenced items (no `rg` hits outside the files themselves, last modified within the cutoff):
    - `agents/codex-3685.md` (last touched 2025-11-19) – candidate for archival under `archives/docs/` unless re-linked from automation docs.
    - `man/actionlint.1` (last touched 2025-11-14) – unused manual page; consider relocating to `archives/man/` or removing if actionlint is covered elsewhere.

## Data/log outputs to retain vs. archive
- `reports/`: keep `reports/tearsheet.md` as the representative report; no older reports present to archive.
- `perf/`: keep `perf/perf_baseline.json`; no additional baselines present to stage for archiving.
- `analysis/`: directory only contains source modules (`__init__.py`, `cv.py`, `results.py`, `tearsheet.py`); no generated outputs to archive.
- `notebooks/`: keep `notebooks/Vol_Adj_Trend_Analysis1.5.TrEx.ipynb`; no legacy notebooks present to stage for archiving.

## Duplicated or overlapping docs
- Quickstart coverage currently spans `README.md` (CLI-centric quickstart), `docs/quickstart.md` (Docker + local setup for end users), and `DOCKER_QUICKSTART.md` (Docker build/run details). Consolidate by keeping `docs/quickstart.md` as the end-user guide, trimming `README.md` to a short pointer for setup, and reducing `DOCKER_QUICKSTART.md` to advanced build notes linked from the main quickstart.
