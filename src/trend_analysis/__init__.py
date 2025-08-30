"""Trend analysis package."""

import importlib.metadata

from . import metrics, config, data, pipeline, export, selector, weighting
from .data import load_csv, identify_risk_free_fund
from .export import (
    register_formatter_excel,
    reset_formatters_excel,
    make_summary_formatter,
    export_to_excel,
    export_to_csv,
    export_to_json,
    export_to_txt,
    export_data,
    metrics_from_result,
    combined_summary_result,
    combined_summary_frame,
    phase1_workbook_data,
    flat_frames_from_results,
    export_phase1_workbook,
    export_phase1_multi_metrics,
    export_multi_period_metrics,
    export_bundle,
)

# Expose multi-period CLI
from . import run_multi_analysis

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("trend-analysis")
except importlib.metadata.PackageNotFoundError:
    # Fallback version for development
    __version__ = "0.1.0-dev"

__all__ = [
    "metrics",
    "config",
    "data",
    "pipeline",
    "export",
    "selector",
    "weighting",
    "load_csv",
    "identify_risk_free_fund",
    "register_formatter_excel",
    "reset_formatters_excel",
    "make_summary_formatter",
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_to_txt",
    "export_data",
    "metrics_from_result",
    "combined_summary_result",
    "combined_summary_frame",
    "phase1_workbook_data",
    "flat_frames_from_results",
    "export_phase1_workbook",
    "export_phase1_multi_metrics",
    "export_multi_period_metrics",
    "export_bundle",
    "run_multi_analysis",
    "__version__",
]
