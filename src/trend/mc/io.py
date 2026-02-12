from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


class MCNavPathsIOError(RuntimeError):
    """Raised when Monte Carlo NAV-path inputs are missing or invalid."""


MISSING_NAV_PATHS_RAISE = "raise"
MISSING_NAV_PATHS_RETURN_NONE = "return-none"
SUPPORTED_MISSING_NAV_PATHS_BEHAVIORS = frozenset(
    {MISSING_NAV_PATHS_RAISE, MISSING_NAV_PATHS_RETURN_NONE}
)


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


def validate_nav_paths_df(
    nav_paths_df: object, *, required_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Validate a loaded NAV-path frame."""

    if not isinstance(nav_paths_df, pd.DataFrame):
        raise MCNavPathsIOError("NAV paths data must be a pandas DataFrame.")
    if nav_paths_df.empty:
        raise MCNavPathsIOError("NAV paths DataFrame must not be empty.")
    if required_columns is not None:
        required = tuple(required_columns)
        missing = sorted(str(col) for col in required if col not in nav_paths_df.columns)
        if missing:
            missing_text = ", ".join(missing)
            raise MCNavPathsIOError(
                f"NAV paths DataFrame is missing required column(s): {missing_text}"
            )
    return nav_paths_df


def load_nav_paths(
    bundle: str | os.PathLike[str],
    *,
    missing_parquet: str = MISSING_NAV_PATHS_RETURN_NONE,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    """Load ``nav_paths.parquet`` from an MC bundle directory.

    Parameters
    ----------
    bundle:
        Filesystem path to the MC bundle directory.
    missing_parquet:
        Behavior when ``nav_paths.parquet`` is absent. Supported values are
        ``"return-none"`` and ``"raise"``.
    required_columns:
        Optional required columns enforced by ``validate_nav_paths_df``.

    Returns
    -------
    pd.DataFrame | None
        Parsed NAV paths table when ``nav_paths.parquet`` exists. Returns ``None`` only
        when ``missing_parquet="return-none"`` and the parquet file is absent.

    Raises
    ------
    MCNavPathsIOError
        If the bundle path is invalid, unsupported nav_paths file formats are detected,
        parquet loading fails, validation fails, or ``missing_parquet="raise"`` and the
        parquet file is absent.
    """

    if missing_parquet not in SUPPORTED_MISSING_NAV_PATHS_BEHAVIORS:
        allowed = ", ".join(sorted(SUPPORTED_MISSING_NAV_PATHS_BEHAVIORS))
        raise MCNavPathsIOError(
            f"Unsupported missing_parquet behavior '{missing_parquet}'. Supported values: {allowed}"
        )

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
        if missing_parquet == MISSING_NAV_PATHS_RAISE:
            raise MCNavPathsIOError(
                f"Missing required nav_paths.parquet file in MC bundle: {bundle_dir}"
            )
        return None
    loaded = _read_nav_paths_parquet(nav_paths_path)
    return validate_nav_paths_df(loaded, required_columns=required_columns)


def load_nav_paths_frame(bundle: str | os.PathLike[str]) -> pd.DataFrame | None:
    """Backward-compatible wrapper for optional NAV-path loading."""

    return load_nav_paths(bundle, missing_parquet=MISSING_NAV_PATHS_RETURN_NONE)


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
