"""Trend analysis package."""

import importlib
import importlib.metadata
from typing import Any

# Attempt to import submodules.  Some of them pull in optional heavy
# dependencies (e.g. ``matplotlib``).  Missing extras shouldn't prevent
# lightweight helpers such as ``io.validators`` from being imported, so we
# ignore ``ModuleNotFoundError`` during initialisation.
_SUBMODULES = [
    "api",
    "metrics",
    "config",
    "data",
    "pipeline",
    "export",
    "selector",
    "weighting",
    "run_multi_analysis",
]

for _name in _SUBMODULES:
    try:
        globals()[_name] = importlib.import_module(f"trend_analysis.{_name}")
    except ModuleNotFoundError as e:
        # Only suppress if the missing module is NOT the submodule itself
        if e.name == f"{__name__}.{_name}":
            raise
        # Optional dependency for this submodule is missing; skip exposing it.
        pass

# Forward declarations for static type checkers; actual values are assigned
# dynamically above via importlib. This avoids mypy complaints about names
# listed in __all__ not being present in the module at type-check time.
metrics: Any
config: Any
data: Any
pipeline: Any
export: Any
selector: Any
weighting: Any
run_multi_analysis: Any
api: Any

if "data" in globals():
    # Conditional import: 'data' submodule may not always be present
    # due to optional dependencies.
    from .data import identify_risk_free_fund, load_csv

if "export" in globals():
    from .export import (
        combined_summary_frame,
        combined_summary_result,
        export_bundle,
        export_data,
        export_multi_period_metrics,
        export_phase1_multi_metrics,
        export_phase1_workbook,
        export_to_csv,
        export_to_excel,
        export_to_json,
        export_to_txt,
        flat_frames_from_results,
        make_summary_formatter,
        metrics_from_result,
        phase1_workbook_data,
        register_formatter_excel,
        reset_formatters_excel,
    )

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
    "api",
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
