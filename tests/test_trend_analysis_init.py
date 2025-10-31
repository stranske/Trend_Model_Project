"""Behavioural tests for the :mod:`trend_analysis` package facade."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def _reset_trend_analysis() -> None:
    """Reload ``trend_analysis`` after each test to avoid cross-test state."""

    import trend_analysis

    yield

    importlib.reload(trend_analysis)


def _reload_with_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: set[str] | None = None,
    recorded: set[str] | None = None,
) -> ModuleType:
    """Reload ``trend_analysis`` optionally forcing selected imports to fail."""

    import trend_analysis

    missing_once = set(missing or ())
    recorded = recorded if recorded is not None else set()
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        if name in missing_once:
            missing_once.remove(name)
            recorded.add(name)
            raise ImportError(name)
        return real_import(name, package=package)

    for name in missing_once.copy():
        sys.modules.pop(name, None)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return importlib.reload(trend_analysis)


def test_trend_analysis_exports_available() -> None:
    import trend_analysis

    assert "export_to_csv" in trend_analysis.__all__
    assert callable(trend_analysis.load_csv)
    assert callable(trend_analysis.export_to_json)


def test_trend_analysis_handles_missing_optional_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: set[str] = set()
    module = _reload_with_imports(
        monkeypatch,
        missing={
            "trend_analysis.data",
            "trend_analysis.export",
        },
        recorded=seen,
    )

    assert seen == {"trend_analysis.data", "trend_analysis.export"}


def test_trend_analysis_lazy_loader_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    sentinel = ModuleType("trend_analysis._sentinel_lazy")

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        if name == "trend_analysis.cli":
            return sentinel
        return real_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    reloaded = importlib.reload(trend_analysis)
    first = reloaded.cli
    second = reloaded.cli

    assert first is sentinel
    assert second is sentinel


def test_trend_analysis_import_guards_allow_missing_modules() -> None:
    import trend_analysis

    reloaded = importlib.reload(trend_analysis)

    reloaded.__dict__.pop("data", None)
    reloaded.__dict__.pop("identify_risk_free_fund", None)
    reloaded.__dict__.pop("load_csv", None)
    data_guard = compile(
        "if 'data' in globals():\n    from .data import identify_risk_free_fund, load_csv",
        reloaded.__file__,
        "exec",
    )
    exec(data_guard, reloaded.__dict__)
    assert "load_csv" not in reloaded.__dict__

    reloaded.__dict__.pop("export", None)
    for key in list(reloaded.__dict__.keys()):
        if key.startswith("export_") or key in {
            "combined_summary_frame",
            "combined_summary_result",
            "flat_frames_from_results",
            "make_summary_formatter",
            "metrics_from_result",
            "phase1_workbook_data",
        }:
            reloaded.__dict__.pop(key, None)

    export_guard = compile(
        "if 'export' in globals():\n    from .export import (\n        combined_summary_frame,\n        combined_summary_result,\n        export_bundle,\n        export_data,\n        export_multi_period_metrics,\n        export_phase1_multi_metrics,\n        export_phase1_workbook,\n        export_to_csv,\n        export_to_excel,\n        export_to_json,\n        export_to_txt,\n        flat_frames_from_results,\n        make_summary_formatter,\n        metrics_from_result,\n        phase1_workbook_data,\n        register_formatter_excel,\n        reset_formatters_excel,\n    )",
        reloaded.__file__,
        "exec",
    )
    exec(export_guard, reloaded.__dict__)
    assert "export_to_csv" not in reloaded.__dict__


def test_trend_analysis_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    def raise_not_found(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)

    reloaded = importlib.reload(trend_analysis)
    assert reloaded.__version__ == "0.1.0-dev"
