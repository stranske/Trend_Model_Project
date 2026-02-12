from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


class MCNavPathsIOError(RuntimeError):
    """Raised when Monte Carlo NAV-path inputs are missing or invalid."""


def _read_nav_paths_parquet(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise MCNavPathsIOError(
            f"Failed to read nav_paths parquet file '{path.name}'. "
            "The file may be corrupted or not a parquet file."
        ) from exc
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise MCNavPathsIOError(f"Expected nav_paths data in '{path}' to load as a table.")
    return frame


def load_nav_paths_frame(bundle: str | os.PathLike[str]) -> pd.DataFrame | None:
    """Load optional ``nav_paths.parquet`` from an MC bundle directory.

    Parameters
    ----------
    bundle:
        Filesystem path to the MC bundle directory.

    Returns
    -------
    pd.DataFrame | None
        Parsed NAV paths table when ``nav_paths.parquet`` exists, otherwise ``None``.

    Raises
    ------
    MCNavPathsIOError
        If the bundle path is invalid, unsupported nav_paths file formats are detected,
        or parquet loading fails.
    """

    bundle_dir = Path(bundle).expanduser().resolve()
    if not bundle_dir.exists():
        raise MCNavPathsIOError(f"MC bundle directory does not exist: {bundle_dir}")
    if not bundle_dir.is_dir():
        raise MCNavPathsIOError(f"MC bundle path is not a directory: {bundle_dir}")

    nav_paths_path = bundle_dir / "nav_paths.parquet"
    if not nav_paths_path.exists():
        unsupported_paths = tuple(bundle_dir / f"nav_paths.{ext}" for ext in ("csv", "json"))
        detected_unsupported = [path for path in unsupported_paths if path.exists()]
        if detected_unsupported:
            unsupported_text = ", ".join(
                f"'{path.name}' ({path.suffix.lower()})" for path in detected_unsupported
            )
            raise MCNavPathsIOError(
                "Unsupported nav_paths file format(s) detected in MC bundle: "
                f"{unsupported_text}. Only nav_paths.parquet is supported."
            )
        return None
    return _read_nav_paths_parquet(nav_paths_path)


def validate_nav_paths_requirement(
    selected_charts: Iterable[str],
    nav_paths_frame: pd.DataFrame | None,
    *,
    nav_path_required_charts: set[str] | frozenset[str],
) -> None:
    """Validate chart selection against NAV-path data availability.

    Parameters
    ----------
    selected_charts:
        Requested chart identifiers from CLI input.
    nav_paths_frame:
        Loaded NAV-path data; ``None`` indicates ``nav_paths.parquet`` is absent.
    nav_path_required_charts:
        Chart IDs that require NAV-path inputs.

    Raises
    ------
    MCNavPathsIOError
        If any selected chart requires NAV-path data but ``nav_paths.parquet`` is missing.
    """

    if nav_paths_frame is not None:
        return
    required = sorted(set(selected_charts).intersection(nav_path_required_charts))
    if not required:
        return
    chart_text = ", ".join(required)
    raise MCNavPathsIOError(
        f"Chart(s) {chart_text} require nav_paths.parquet in the MC bundle. "
        "Add nav_paths.parquet or remove these chart(s) from --charts."
    )
