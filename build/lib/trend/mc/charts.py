"""Chart-selection constants for Monte Carlo visualization."""

from __future__ import annotations

# Charts that require nav_paths.parquet inputs.
NAV_PATH_REQUIRED_CHARTS: frozenset[str] = frozenset({"path_dist"})

__all__ = ["NAV_PATH_REQUIRED_CHARTS"]
