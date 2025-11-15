"""Shared report generation utilities for CLI and Streamlit layers."""

from __future__ import annotations

from .quick_summary import build_run_report, render_parameter_grid_heatmap
from .unified import ReportArtifacts, generate_unified_report

__all__ = [
    "ReportArtifacts",
    "build_run_report",
    "generate_unified_report",
    "render_parameter_grid_heatmap",
]
