"""Trend analysis package."""

import importlib
import importlib.metadata
import sys
from types import ModuleType
from typing import Any, cast


def _patch_dataclasses_module_guard() -> None:
    """Ensure dataclass processing tolerates cleared ``sys.modules`` entries.

    Some heavy integration tests mutate ``sys.modules`` by removing previously
    imported ``tests.*`` packages.  When later tests define dataclasses within
    those modules, the stdlib ``dataclasses`` helper attempts to look the module
    back up and crashes when it is absent.  We patch the private
    ``dataclasses._is_type`` helper so it re-imports the missing module (falling
    back to a lightweight placeholder) before retrying the lookup.  The patch is
    safe for production code because it only triggers when the module reference
    truly disappeared, which should not happen during normal execution.
    """

    import dataclasses

    original = getattr(dataclasses, "_is_type", None)
    if original is None:
        return
    if getattr(dataclasses, "_trend_model_patched", False):
        globals()["_SAFE_IS_TYPE"] = dataclasses._is_type  # type: ignore[attr-defined]
        return

    def _safe_is_type(
        annotation: Any,
        cls: type[Any],
        a_module: Any,
        a_type: Any,
        predicate: Any,
    ) -> bool:
        try:
            return bool(original(annotation, cls, a_module, a_type, predicate))
        except AttributeError:
            module_name = getattr(cls, "__module__", None)
            if not module_name:
                raise

            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    module = ModuleType(module_name)
                    module.__dict__["__package__"] = module_name.rpartition(".")[0]
                sys.modules[module_name] = module

            return bool(original(annotation, cls, a_module, a_type, predicate))

    dataclasses._is_type = _safe_is_type  # type: ignore[attr-defined]
    dataclasses._trend_model_patched = True  # type: ignore[attr-defined]
    globals()["_SAFE_IS_TYPE"] = _safe_is_type


_patch_dataclasses_module_guard()

_MODULE_SELF = sys.modules[__name__]


def _ensure_registered() -> None:
    if sys.modules.get(__name__) is not _MODULE_SELF:
        sys.modules[__name__] = _MODULE_SELF


_ORIGINAL_SPEC = globals().get("__spec__")


class _SpecProxy:
    __slots__ = ("_spec",)

    def __init__(self, spec: Any) -> None:
        self._spec = spec

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._spec, attr)

    @property
    def name(self) -> str:
        _ensure_registered()
        return cast(str, getattr(self._spec, "name"))


if _ORIGINAL_SPEC is not None:
    globals()["__spec__"] = _SpecProxy(_ORIGINAL_SPEC)

_ensure_registered()

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
    "io": "trend_analysis.io",
    "selector": "trend_analysis.selector",
    "weighting": "trend_analysis.weighting",
    "weights": "trend_analysis.weights",
    "presets": "trend_analysis.presets",
    "run_multi_analysis": "trend_analysis.run_multi_analysis",
    "engine": "trend_analysis.engine",
    "perf": "trend_analysis.perf",
    "regimes": "trend_analysis.regimes",
    "multi_period": "trend_analysis.multi_period",
    "plugins": "trend_analysis.plugins",
    "proxy": "trend_analysis.proxy",
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
io: Any
backtesting: Any
api: Any
selector: Any
weighting: Any
presets: Any
run_multi_analysis: Any
engine: Any
perf: Any
regimes: Any
multi_period: Any
plugins: Any
proxy: Any

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
    "io",
    "signals",
    "presets",
    "selector",
    "weighting",
    "plugins",
    "proxy",
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
    "engine",
    "perf",
    "regimes",
    "multi_period",
    "__version__",
]
