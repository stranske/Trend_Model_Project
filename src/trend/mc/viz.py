"""Shared implementation for ``trend mc viz`` command execution."""

from __future__ import annotations

import importlib.util
import sys
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
    """Return ``True`` when the Plotly PNG-export dependency is usable.

    Checks both that kaleido is importable *and* that it can actually
    produce a PNG (kaleido 1.x may import but fail with newer Plotly).
    """
    try:
        if importlib.util.find_spec("kaleido") is None:
            return False
        import plotly.io as pio

        # Use a minimal figure dict (not `go.Figure`) to avoid triggering
        # template initialization during dependency checks.
        fig = {"data": [{"type": "scatter", "x": [0, 1], "y": [0, 1]}]}
        pio.to_image(fig, format="png", validate=True)
        return True
    except Exception:  # noqa: BLE001 – broad catch intentional
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_chart_ids(charts: Sequence[str] | str) -> list[str]:
    if isinstance(charts, str):
        normalized = [
            part.strip().lower() for part in charts.split(",") if part.strip()
        ]
    else:
        normalized = [str(part).strip().lower() for part in charts if str(part).strip()]
    return normalized


def _collect_required_inputs(charts: Sequence[str] | str) -> list[str]:
    requested_charts = _normalize_chart_ids(charts)
    unsupported = [
        chart for chart in requested_charts if chart not in CHART_REQUIREMENTS
    ]
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
        raise TrendCLIError(
            f"Failed to read {label} data from '{path}': {exc}"
        ) from exc
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise TrendCLIError(f"Expected {label} data in '{path}' to load as a table.")
    return frame


def _load_mc_frame(bundle_dir: Path, *, stem: str) -> pd.DataFrame:
    candidates = tuple(
        bundle_dir / f"{stem}.{ext}" for ext in _OPTIONAL_STEM_EXTENSIONS
    )
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


def _available_fold_nav_paths(bundle_dir: Path) -> list[Path]:
    return sorted(bundle_dir.glob("nav_paths_fold_*.parquet"))


def _load_fold_nav_paths_frame(bundle_dir: Path, fold_id: int) -> pd.DataFrame:
    nav_path = bundle_dir / f"nav_paths_fold_{fold_id}.parquet"
    if not nav_path.exists():
        available = _available_fold_nav_paths(bundle_dir)
        available_text = ", ".join(p.name for p in available) if available else "(none)"
        raise TrendCLIError(
            f"Requested --fold {fold_id} but '{nav_path.name}' was not found in the bundle. "
            f"Available fold nav paths: {available_text}."
        )
    return _read_mc_frame(nav_path, label=f"nav_paths_fold_{fold_id}")


def _load_nav_paths_override(nav_paths: str | Path) -> pd.DataFrame:
    nav_path = Path(nav_paths).expanduser().resolve()
    if not nav_path.exists():
        raise TrendCLIError(f"Requested --nav-paths file does not exist: {nav_path}")
    if not nav_path.is_file():
        raise TrendCLIError(f"Requested --nav-paths path is not a file: {nav_path}")
    if nav_path.suffix.lower() != ".parquet":
        raise TrendCLIError(
            f"Unsupported --nav-paths format '{nav_path.suffix}' for '{nav_path.name}'. "
            "Only parquet files are supported."
        )
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
        returns_frame = nav_frame.apply(pd.to_numeric, errors="coerce").dropna(
            how="all"
        )
    return risk_return.make(returns_frame)


def _mc_chart_builders() -> dict[str, Any]:
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
    *,
    fold_id: int | None,
    nav_paths: str | Path | None,
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
        validate_nav_paths_df,
        validate_nav_paths_requirement,
    )

    try:
        bundle_dir = Path(bundle_path).expanduser().resolve()
        if nav_paths is not None:
            nav_paths_frame = validate_nav_paths_df(_load_nav_paths_override(nav_paths))
        elif fold_id is not None:
            nav_paths_frame = validate_nav_paths_df(
                _load_fold_nav_paths_frame(bundle_dir, fold_id)
            )
        else:
            nav_paths_frame = load_nav_paths_frame(bundle_path)
            if nav_paths_frame is None and set(selected_charts).intersection(
                NAV_PATH_REQUIRED_CHARTS
            ):
                fold_paths = _available_fold_nav_paths(bundle_dir)
                if fold_paths:
                    names = ", ".join(p.name for p in fold_paths[:5])
                    if len(fold_paths) > 5:
                        names = f"{names}, ..."
                    raise TrendCLIError(
                        "nav_paths.parquet is missing, but fold-exported NAV-path files were found "
                        f"({names}). Re-run with --fold <id> to select a fold, or provide "
                        "--nav-paths <file> to point at a specific nav_paths parquet file."
                    )
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

    Public API contract
    -------------------
    The parameter list for this function is intentionally stable because
    downstream tooling (and tests) relies on it.

    For CLI-only extensions (for example fold-nav-path selection), prefer
    calling :func:`execute_mc_viz_cli`.
    """

    return execute_mc_viz_cli(
        bundle_path=bundle_path,
        out_dir=out_dir,
        charts=charts,
        fold_id=None,
        nav_paths=None,
        html=html,
        json=json,
        png=png,
    )


def execute_mc_viz_cli(
    bundle_path: str | Path,
    out_dir: str | Path,
    charts: Sequence[str] | str,
    *,
    fold_id: int | None,
    nav_paths: str | Path | None,
    html: bool,
    json: bool,
    png: bool,
) -> int:
    """CLI-oriented Monte Carlo viz execution.

    This entry point extends :func:`execute_mc_viz` with optional helpers used
    by the ``trend mc viz`` CLI (for example selecting fold-exported NAV paths).
    """
    # -- Output flag validation ------------------------------------------------
    if not any((html, json, png)):
        raise TrendCLIError(
            "The 'mc viz' command requires at least one output flag: "
            "--html, --json, or --png"
        )

    # -- PNG dependency check – degrade gracefully ----------------------------
    if png and not check_png_dependency():
        png = False
        print(
            "PNG export skipped: the kaleido package is missing or "
            "incompatible with the installed Plotly version. "
            "Install a compatible kaleido: pip install kaleido",
            file=sys.stderr,
        )
        # Re-check: if png was the only requested format, fail early.
        if not any((html, json, png)):
            raise TrendCLIError(
                "PNG export requires a working kaleido installation and no "
                "other output format was requested. "
                "Install kaleido: pip install kaleido"
            )

    # -- Load bundle frames ----------------------------------------------------
    summary_frame, results_frame = _load_mc_bundle_frames(bundle_path)
    selected_charts = _parse_mc_chart_selection(charts)

    # -- Nav-paths validation --------------------------------------------------
    nav_paths_frame, uses_fallback_nav_data = _validate_nav_paths(
        bundle_path,
        selected_charts,
        fold_id=fold_id,
        nav_paths=nav_paths,
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

    # -- HTML chart markers ----------------------------------------------------
    if html:
        _inject_mc_html_chart_markers(plots_dir, selected_charts, warnings=warnings)

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
            print(
                "PNG export skipped: no PNG files were produced. "
                "The kaleido package may be incompatible with the "
                "installed version of Plotly.",
                file=sys.stderr,
            )

    return 0


def _inject_mc_html_chart_markers(
    plots_dir: Path,
    selected_charts: list[str],
    *,
    warnings: list[str],
) -> None:
    """Inject deterministic HTML markers into chart HTML files."""
    for chart_id in selected_charts:
        html_path = plots_dir / f"{chart_id}.html"
        if not html_path.exists():
            continue
        marker = f"<!-- mc-viz-chart:{chart_id} -->"
        try:
            html_text = html_path.read_text(encoding="utf-8")
            if marker in html_text:
                continue
            body_token = "<body>"
            if body_token in html_text:
                updated_html = html_text.replace(
                    body_token, f"{body_token}\n{marker}", 1
                )
            else:
                updated_html = f"{marker}\n{html_text}"
            html_path.write_text(updated_html, encoding="utf-8")
        except Exception as exc:
            warnings.append(
                f"Unable to inject HTML chart marker for '{chart_id}': {exc}."
            )


__all__ = [
    "CHART_REQUIREMENTS",
    "NAV_PATH_REQUIRED_CHARTS",
    "TrendCLIError",
    "check_png_dependency",
    "execute_mc_viz",
    "validate_mc_viz_bundle_requirements",
]
