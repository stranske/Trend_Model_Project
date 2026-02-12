"""Shared implementation for ``trend mc viz`` command execution."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_OPTIONAL_STEM_EXTENSIONS: tuple[str, ...] = ("parquet", "csv", "json")

NAV_PATH_REQUIRED_CHARTS: frozenset[str] = frozenset({"path_dist"})

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


class TrendCLIError(RuntimeError):
    """Raised when CLI validation fails before dispatching work."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_mc_viz_bundle_requirements(
    bundle_path: str | Path, charts: Sequence[str] | str
) -> list[str]:
    """Return missing bundle requirements for the requested chart set.

    Parameters
    ----------
    bundle_path
        Path to the Monte Carlo bundle directory.
    charts
        Requested chart IDs as either a comma-separated string or sequence.

    Returns
    -------
    list[str]
        Missing requirement labels in deterministic order.
    """
    bundle_dir = Path(bundle_path).expanduser().resolve()
    requirements = _collect_required_inputs(charts)
    return [
        _missing_requirement_label(requirement)
        for requirement in requirements
        if not _requirement_is_present(bundle_dir, requirement)
    ]


def check_png_dependency() -> bool:
    """Return ``True`` when the Plotly PNG-export dependency is usable."""
    try:
        return importlib.util.find_spec("kaleido") is not None
    except (ModuleNotFoundError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_chart_ids(charts: Sequence[str] | str) -> list[str]:
    if isinstance(charts, str):
        normalized = [part.strip().lower() for part in charts.split(",") if part.strip()]
    else:
        normalized = [str(part).strip().lower() for part in charts if str(part).strip()]
    return normalized


def _collect_required_inputs(charts: Sequence[str] | str) -> list[str]:
    requested_charts = _normalize_chart_ids(charts)
    unsupported = [chart for chart in requested_charts if chart not in CHART_REQUIREMENTS]
    if unsupported:
        invalid = ", ".join(sorted(set(unsupported)))
        raise ValueError(f"Unsupported chart identifier(s): {invalid}")

    requirements: list[str] = []
    seen: set[str] = set()
    for chart in requested_charts:
        for requirement in CHART_REQUIREMENTS[chart]:
            if requirement not in seen:
                seen.add(requirement)
                requirements.append(requirement)
    return requirements


def _requirement_is_present(bundle_dir: Path, requirement: str) -> bool:
    if "." in requirement:
        return (bundle_dir / requirement).exists()
    return any(
        (bundle_dir / f"{requirement}.{ext}").exists()
        for ext in _OPTIONAL_STEM_EXTENSIONS
    )


def _missing_requirement_label(requirement: str) -> str:
    if "." in requirement:
        return requirement
    options = "/".join(f"{requirement}.{ext}" for ext in _OPTIONAL_STEM_EXTENSIONS)
    return f"{options} (one required)"


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


def _read_mc_frame(path: Path, *, label: str) -> pd.DataFrame:
    """Read a single data frame from *path*, raising ``TrendCLIError`` on failure."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                raise TrendCLIError(
                    f"Failed to read {label} parquet file '{path.name}'. "
                    "The file may be corrupted or not a parquet file."
                ) from exc
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            raise TrendCLIError(
                f"Unsupported {label} file format '{path.suffix}' for '{path.name}'."
            )
    except TrendCLIError:
        raise
    except Exception as exc:
        raise TrendCLIError(f"Failed to read {label} data from '{path}': {exc}") from exc
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise TrendCLIError(f"Expected {label} data in '{path}' to load as a table.")
    return frame


def _load_mc_frame(bundle_dir: Path, *, stem: str) -> pd.DataFrame:
    candidates = tuple(bundle_dir / f"{stem}.{ext}" for ext in _OPTIONAL_STEM_EXTENSIONS)
    existing = next((c for c in candidates if c.exists()), None)
    if existing is None:
        expected = ", ".join(p.name for p in candidates)
        raise TrendCLIError(
            f"Missing required MC {stem} file in '{bundle_dir}'. Expected one of: {expected}"
        )
    return _read_mc_frame(existing, label=stem)


def _load_mc_bundle_frames(bundle: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle_dir = Path(bundle).expanduser().resolve()
    if not bundle_dir.exists():
        raise TrendCLIError(f"MC bundle directory does not exist: {bundle_dir}")
    if not bundle_dir.is_dir():
        raise TrendCLIError(f"MC bundle path is not a directory: {bundle_dir}")

    required_stems = ("summary", "results")
    missing_inputs: list[str] = []
    expected_by_stem: dict[str, str] = {}
    for stem in required_stems:
        candidates = tuple(
            bundle_dir / f"{stem}.{ext}" for ext in _OPTIONAL_STEM_EXTENSIONS
        )
        if not any(c.exists() for c in candidates):
            missing_inputs.append(stem)
            expected_by_stem[stem] = ", ".join(p.name for p in candidates)
    if missing_inputs:
        if len(missing_inputs) == 1:
            stem = missing_inputs[0]
            expected = expected_by_stem[stem]
            raise TrendCLIError(
                f"Missing required MC {stem} file in '{bundle_dir}'. "
                f"Expected one of: {expected}"
            )
        missing_text = ", ".join(missing_inputs)
        expected_text = "; ".join(
            f"{stem}: {expected_by_stem[stem]}" for stem in missing_inputs
        )
        raise TrendCLIError(
            f"Missing required MC input files in '{bundle_dir}': {missing_text}. "
            f"Expected one of each: {expected_text}"
        )
    summary = _load_mc_frame(bundle_dir, stem="summary")
    results = _load_mc_frame(bundle_dir, stem="results")
    return summary, results


def _load_mc_nav_paths_frame(bundle: str | Path) -> pd.DataFrame | None:
    bundle_dir = Path(bundle).expanduser().resolve()
    nav_path = bundle_dir / "nav_paths.parquet"
    if not nav_path.exists():
        return None
    return _read_mc_frame(nav_path, label="nav_paths")


# ---------------------------------------------------------------------------
# Chart parsing / selection
# ---------------------------------------------------------------------------


def _parse_mc_chart_selection(charts_value: str | Sequence[str]) -> list[str]:
    if isinstance(charts_value, str):
        requested = [t.strip().lower() for t in charts_value.split(",") if t.strip()]
    else:
        requested = [str(t).strip().lower() for t in charts_value if str(t).strip()]
    if not requested:
        raise TrendCLIError(
            "The 'mc viz' command requires at least one chart in --charts."
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for chart in requested:
        if chart not in seen:
            seen.add(chart)
            ordered.append(chart)

    supported = tuple(_mc_chart_builders().keys())
    unsupported = [c for c in ordered if c not in supported]
    if unsupported:
        supported_text = ", ".join(supported)
        invalid_text = ", ".join(unsupported)
        raise TrendCLIError(
            f"Unsupported chart identifier(s): {invalid_text}. "
            f"Supported charts: {supported_text}"
        )
    return ordered


# ---------------------------------------------------------------------------
# Nav source derivation
# ---------------------------------------------------------------------------


def _mc_nav_source_frame(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if nav_paths_frame is not None:
        return nav_paths_frame
    for frame in (results_frame, summary_frame):
        numeric = frame.select_dtypes(include=[np.number]).copy()
        numeric = numeric.dropna(how="all")
        if not numeric.empty:
            return numeric
    raise TrendCLIError(
        "Unable to derive path data for Monte Carlo charts. "
        "Provide nav_paths.parquet or numeric summary/results files."
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_mc_fan_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import fan

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    return fan.make(nav_frame)


def _build_mc_path_dist_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import path_dist

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    return path_dist.make(nav_frame)


def _build_mc_risk_return_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import risk_return

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    returns_frame = nav_frame.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    returns_frame = returns_frame.dropna(how="all")
    if returns_frame.empty:
        returns_frame = nav_frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return risk_return.make(returns_frame)


def _mc_chart_builders() -> (
    dict[str, Any]
):
    return {
        "fan": _build_mc_fan_chart,
        "path_dist": _build_mc_path_dist_chart,
        "risk_return": _build_mc_risk_return_chart,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_mc_chart_artifacts(
    charts: Mapping[str, Any],
    out_dir: Path,
    *,
    include_html: bool,
    include_json: bool,
    include_png: bool,
) -> tuple[Path, list[str]]:
    from trend_analysis.monte_carlo.export_bundle import save as export_bundle
    from trend_analysis.viz.artifacts import extract_bundle_zip

    plots_dir = out_dir.expanduser().resolve() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = plots_dir / "mc_charts_bundle.zip"

    warnings: list[str] = []
    export_bundle(
        charts,
        destination=bundle_path,
        include_html=include_html,
        include_json=include_json,
        include_png=include_png,
        warnings=warnings,
    )
    extract_bundle_zip(bundle_path, plots_dir, warnings=warnings)
    return plots_dir, warnings


# ---------------------------------------------------------------------------
# Nav-paths requirement checking (uses trend.mc.io)
# ---------------------------------------------------------------------------


def _validate_nav_paths(
    bundle_path: str | Path,
    selected_charts: list[str],
) -> tuple[pd.DataFrame | None, bool]:
    """Load nav_paths and validate against chart requirements.

    Returns
    -------
    tuple
        ``(nav_paths_frame, uses_fallback_nav_data)``
    """
    from trend.mc.io import (
        MCNavPathsIOError,
        load_nav_paths_frame,
        validate_nav_paths_requirement,
    )

    try:
        nav_paths_frame = load_nav_paths_frame(bundle_path)
        validate_nav_paths_requirement(
            selected_charts,
            nav_paths_frame,
            nav_path_required_charts=NAV_PATH_REQUIRED_CHARTS,
        )
    except MCNavPathsIOError as exc:
        raise TrendCLIError(str(exc)) from exc

    return nav_paths_frame, nav_paths_frame is None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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

    This function is the shared API surface that both CLI entry points call
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
    # -- Output flag validation ------------------------------------------------
    if not any((html, json, png)):
        raise TrendCLIError(
            "The 'mc viz' command requires at least one output flag: "
            "--html, --json, or --png"
        )

    # -- PNG dependency early-fail ---------------------------------------------
    if png and not check_png_dependency():
        raise TrendCLIError(
            "PNG export requires the kaleido package. "
            "Install kaleido: pip install kaleido"
        )

    # -- Load bundle frames ----------------------------------------------------
    summary_frame, results_frame = _load_mc_bundle_frames(bundle_path)
    selected_charts = _parse_mc_chart_selection(charts)

    # -- Nav-paths validation --------------------------------------------------
    nav_paths_frame, uses_fallback_nav_data = _validate_nav_paths(
        bundle_path, selected_charts
    )

    # -- Build charts ----------------------------------------------------------
    chart_builders = _mc_chart_builders()
    generated_charts = {
        chart_id: chart_builders[chart_id](
            summary_frame, results_frame, nav_paths_frame
        )
        for chart_id in selected_charts
    }

    # -- Export artifacts -------------------------------------------------------
    plots_dir, warnings = _export_mc_chart_artifacts(
        generated_charts,
        Path(out_dir),
        include_html=html,
        include_json=json,
        include_png=png,
    )

    # -- Console feedback ------------------------------------------------------
    counts = f"summary_rows={len(summary_frame)} results_rows={len(results_frame)}"
    if nav_paths_frame is not None:
        counts = f"{counts} nav_paths_rows={len(nav_paths_frame)}"
    print(f"Loaded MC bundle frames: {counts}")
    if uses_fallback_nav_data:
        nav_dependent_text = ", ".join(sorted(NAV_PATH_REQUIRED_CHARTS))
        print(
            "Warning: nav_paths.parquet is missing; requested charts do not include "
            f"NAV-path-dependent visuals ({nav_dependent_text}). "
            "Continuing with fallback data derived from summary/results; "
            "these fallback visuals may be less accurate or misleading.",
            file=sys.stderr,
        )
    print(f"Wrote MC chart artifacts to: {plots_dir}")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    # -- PNG post-check --------------------------------------------------------
    if png:
        png_files = list(plots_dir.glob("*.png"))
        if not png_files:
            raise TrendCLIError(
                "PNG export was requested but no PNG files were produced. "
                "The kaleido package may be incompatible with the installed "
                "version of Plotly. Install kaleido: pip install kaleido"
            )

    return 0


__all__ = [
    "CHART_REQUIREMENTS",
    "NAV_PATH_REQUIRED_CHARTS",
    "TrendCLIError",
    "check_png_dependency",
    "execute_mc_viz",
    "validate_mc_viz_bundle_requirements",
]
