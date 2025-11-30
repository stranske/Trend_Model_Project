# Archived GitHub Config Files

Configuration files archived from `.github/` folder.

## 2025-11-30-orphaned

- `labeler.yml` - Path-based label configuration that was orphaned (no workflow used it)
  - The README.md referenced a `pr-path-labeler.yml` workflow that doesn't exist
  - To restore: create a workflow using `actions/labeler` that references this config
