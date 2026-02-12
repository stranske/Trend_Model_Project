"""Shared implementation for ``trend mc viz`` command execution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


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


__all__ = ["execute_mc_viz"]
