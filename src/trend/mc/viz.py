"""Shared implementation for ``trend mc viz`` command execution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_OPTIONAL_STEM_EXTENSIONS: tuple[str, ...] = ("parquet", "csv", "json")

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
    return any((bundle_dir / f"{requirement}.{ext}").exists() for ext in _OPTIONAL_STEM_EXTENSIONS)


def _missing_requirement_label(requirement: str) -> str:
    if "." in requirement:
        return requirement
    options = "/".join(f"{requirement}.{ext}" for ext in _OPTIONAL_STEM_EXTENSIONS)
    return f"{options} (one required)"


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


__all__ = ["CHART_REQUIREMENTS", "execute_mc_viz", "validate_mc_viz_bundle_requirements"]
