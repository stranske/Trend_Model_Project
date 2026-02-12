"""Shared implementation for ``trend mc viz`` command execution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

CHART_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "fan": ("summary", "results"),
    "path_dist": ("summary", "results", "nav_paths.parquet"),
    "risk_return": ("summary", "results"),
}
"""Bundle input requirements keyed by MC chart identifier.

Requirement values follow these conventions:
- Stem values (for example ``"summary"``) mean one of ``.parquet``, ``.csv``,
  or ``.json`` must be present for that stem.
- Literal filenames (for example ``"nav_paths.parquet"``) are exact, format-
  specific requirements.
"""


def execute_mc_viz(
    bundle_path: str | Path,
    out_dir: str | Path,
    charts: Sequence[str] | str,
    *,
    html: bool,
    json: bool,
    png: bool,
) -> int:
    """Execute Monte Carlo visualization artifact generation.

    This function is the shared API surface that both CLI entry points will call
    for ``mc viz`` execution.

    Parameters
    ----------
    bundle_path
        Filesystem path to the Monte Carlo export bundle directory containing
        required input artifacts.
    out_dir
        Destination output directory where visualization artifacts are written.
    charts
        Requested chart identifiers, provided as either a comma-separated string
        (for CLI compatibility) or a sequence of individual chart IDs.
    html
        When ``True``, generate HTML chart artifacts.
    json
        When ``True``, generate JSON chart artifacts.
    png
        When ``True``, generate PNG chart artifacts.

    Returns
    -------
    int
        Conventional CLI status code where ``0`` represents success.
    """
    _ = (bundle_path, out_dir, charts, html, json, png)
    raise NotImplementedError("Shared mc viz execution is not implemented yet.")


__all__ = ["CHART_REQUIREMENTS", "execute_mc_viz"]
