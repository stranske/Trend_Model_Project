"""Helpers for exporting Plotly chart bundles as ZIP archives."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence

import plotly.io as pio

__all__ = ["kaleido_available", "save", "save_to_tempfile"]


def save(
    charts: Mapping[str, Any] | Iterable[tuple[str, Any]],
    destination: BytesIO | Path | str | None = None,
    *,
    include_json: bool = True,
    include_html: bool = True,
    include_png: bool = False,
    html_include_plotlyjs: str | bool = "cdn",
    warnings: MutableSequence[str] | None = None,
) -> BytesIO | Path:
    """Write Plotly charts into a ZIP bundle.

    The bundle is populated with one JSON file and one HTML file per chart by
    default. When ``destination`` is omitted, an in-memory ``BytesIO`` buffer is
    returned. When ``destination`` is a path-like value, the ZIP is written to
    disk and the resolved path is returned.
    """
    if not include_json and not include_html and not include_png:
        raise ValueError("At least one of include_json/include_html/include_png must be enabled.")

    chart_items = _normalise_charts(charts)
    if not chart_items:
        raise ValueError("At least one chart must be provided.")

    png_enabled = include_png and kaleido_available()
    if include_png and not png_enabled and warnings is not None:
        warnings.append("PNG export skipped: Kaleido is not installed.")

    bundle_target = _resolve_destination(destination)
    with zipfile.ZipFile(bundle_target, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, chart in chart_items:
            stem = _safe_name(name)
            if include_json:
                try:
                    payload = pio.to_json(chart, validate=True, pretty=False)
                    json.loads(payload)
                except Exception as exc:
                    raise RuntimeError(f"Failed to serialize chart '{name}' to JSON.") from exc
                bundle.writestr(f"{stem}.json", payload.encode("utf-8"))
            if include_html:
                try:
                    html = pio.to_html(
                        chart,
                        full_html=True,
                        include_plotlyjs=html_include_plotlyjs,
                        validate=True,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to render chart '{name}' to HTML.") from exc
                bundle.writestr(f"{stem}.html", html.encode("utf-8"))
            if png_enabled:
                try:
                    png_bytes = pio.to_image(chart, format="png", validate=True)
                except Exception as exc:
                    if warnings is not None:
                        warnings.append(f"PNG export failed for '{name}': {exc}.")
                else:
                    bundle.writestr(f"{stem}.png", png_bytes)

    if isinstance(bundle_target, BytesIO):
        bundle_target.seek(0)
        return bundle_target
    return Path(bundle_target)


def save_to_tempfile(
    charts: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    include_json: bool = True,
    include_html: bool = True,
    include_png: bool = False,
    html_include_plotlyjs: str | bool = "cdn",
    suffix: str = ".zip",
    warnings: MutableSequence[str] | None = None,
) -> Path:
    """Create a temporary chart bundle file and return its path."""

    with tempfile.NamedTemporaryFile(prefix="mc_charts_", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    save(
        charts,
        destination=temp_path,
        include_json=include_json,
        include_html=include_html,
        include_png=include_png,
        html_include_plotlyjs=html_include_plotlyjs,
        warnings=warnings,
    )
    return temp_path


def kaleido_available() -> bool:
    """Return whether Kaleido is importable for Plotly image export."""

    return importlib.util.find_spec("kaleido") is not None


def _normalise_charts(
    charts: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    if isinstance(charts, Mapping):
        return [(str(name), chart) for name, chart in charts.items()]
    return [(str(name), chart) for name, chart in charts]


def _resolve_destination(destination: BytesIO | Path | str | None) -> BytesIO | str:
    if destination is None:
        return BytesIO()
    if isinstance(destination, BytesIO):
        destination.seek(0)
        destination.truncate(0)
        return destination
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "chart"
