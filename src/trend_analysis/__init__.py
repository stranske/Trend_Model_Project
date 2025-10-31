"""Trend analysis package."""

import importlib
import importlib.metadata
from types import ModuleType
from typing import Any

# Attempt to import a core set of lighter submodules eagerly. Heavier or
# optional pieces are exposed lazily via __getattr__ to avoid hard failures
# when the environment is only partially initialised (e.g. before venv
# activation). This also prevents transient ModuleNotFoundError masking when
# optional dependencies of those submodules are absent.
_EAGER_SUBMODULES = [
    "metrics",
    "config",
    "data",
    "pipeline",
    "export",
    "signals",
    "backtesting",
]

# Modules that may drag optional / heavy deps; imported on first attribute access.
_LAZY_SUBMODULES = {
    "api": "trend_analysis.api",
    "cli": "trend_analysis.cli",
    "selector": "trend_analysis.selector",
    "weighting": "trend_analysis.weighting",
    "weights": "trend_analysis.weights",
    "run_multi_analysis": "trend_analysis.run_multi_analysis",
}

# Purge stale lazy-loaded attributes so reload() restores deferred imports.
for _lazy_attr in list(_LAZY_SUBMODULES):
    globals().pop(_lazy_attr, None)

for _name in _EAGER_SUBMODULES:
    try:  # pragma: no cover - import side effects
        globals()[_name] = importlib.import_module(f"trend_analysis.{_name}")
    except ImportError:
        # Missing optional dependency chain; submodule simply not exposed.
        continue


def __getattr__(attr: str) -> ModuleType:  # pragma: no cover - thin lazy loader
    target = _LAZY_SUBMODULES.get(attr)
    if target is None:
        raise AttributeError(attr)
    mod = importlib.import_module(target)
    globals()[attr] = mod
    return mod


# Forward declarations for static type checkers; actual values are assigned
# dynamically above via importlib. This avoids mypy complaints about names
# listed in __all__ not being present in the module at type-check time.
metrics: Any
config: Any
data: Any
pipeline: Any
export: Any
signals: Any
backtesting: Any
api: Any
selector: Any
weighting: Any
run_multi_analysis: Any

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
    "backtesting",
    "api",
    "export",
    "signals",
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
