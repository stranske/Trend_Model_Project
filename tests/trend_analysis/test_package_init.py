from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType
from typing import Mapping

import pytest


def _clear_trend_analysis_modules() -> None:
    for name in [
        key for key in sys.modules if key == "trend_analysis" or key.startswith("trend_analysis.")
    ]:
        sys.modules.pop(name, None)


@pytest.fixture
def load_trend_analysis():
    preserved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "trend_analysis" or name.startswith("trend_analysis.")
    }

    def loader(*, preloaded: Mapping[str, ModuleType] | None = None) -> ModuleType:
        _clear_trend_analysis_modules()
        if preloaded:
            for name, module in preloaded.items():
                sys.modules[name] = module
        return importlib.import_module("trend_analysis")

    yield loader

    _clear_trend_analysis_modules()
    for name, module in preserved.items():
        sys.modules[name] = module

    sys.modules.pop("tests.fake_module_for_trend_init", None)
    sys.modules.pop("tests.fake_module_for_trend_init_existing", None)
    sys.modules.pop("tests.fake_module_for_trend_init_retry", None)


def test_import_does_not_mutate_dataclasses(load_trend_analysis):
    import dataclasses

    original_is_type = getattr(dataclasses, "_is_type", None)
    load_trend_analysis()
    assert getattr(dataclasses, "_is_type", None) is original_is_type
    assert not hasattr(dataclasses, "_trend_patched")
    assert not hasattr(dataclasses, "_trend_model_patched")


def test_spec_proxy_restores_module_registration(load_trend_analysis):
    module = load_trend_analysis()
    sys.modules.pop("trend_analysis", None)

    spec = module.__spec__
    assert spec.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is module


def test_lazy_loader_imports_module_on_demand(load_trend_analysis):
    module = load_trend_analysis()
    sys.modules.pop("trend_analysis.cli", None)
    module.__dict__.pop("cli", None)

    lazy_loaded = module.cli

    assert lazy_loaded is sys.modules["trend_analysis.cli"]
    assert module.__dict__["cli"] is lazy_loaded


def test_lazy_loader_rejects_unknown_attribute(load_trend_analysis):
    module = load_trend_analysis()
    with pytest.raises(AttributeError):
        getattr(module, "not_a_real_submodule")


def test_version_metadata_success_path(monkeypatch, load_trend_analysis):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    module = load_trend_analysis()
    assert module.__version__ == "9.9.9"


def test_version_metadata_fallback_path(monkeypatch, load_trend_analysis):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda *_: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )
    module = load_trend_analysis()
    assert module.__version__ == "0.1.0-dev"


def test_eager_imports_populate_conditional_exports(load_trend_analysis):
    sentinel = ModuleType("trend_analysis.data")
    sentinel.identify_risk_free_fund = object()
    sentinel.load_csv = object()

    exporter = ModuleType("trend_analysis.export")
    exporter.combined_summary_frame = object()
    exporter.combined_summary_result = object()
    exporter.export_bundle = object()
    exporter.export_data = object()
    exporter.export_multi_period_metrics = object()
    exporter.export_phase1_multi_metrics = object()
    exporter.export_phase1_workbook = object()
    exporter.export_to_csv = object()
    exporter.export_to_excel = object()
    exporter.export_to_json = object()
    exporter.export_to_txt = object()
    exporter.flat_frames_from_results = object()
    exporter.make_summary_formatter = object()
    exporter.metrics_from_result = object()
    exporter.phase1_workbook_data = object()
    exporter.register_formatter_excel = object()
    exporter.reset_formatters_excel = object()

    module = load_trend_analysis(
        preloaded={
            "trend_analysis.data": sentinel,
            "trend_analysis.export": exporter,
        }
    )
    assert module.load_csv is sentinel.load_csv
    assert module.identify_risk_free_fund is sentinel.identify_risk_free_fund
    assert module.export_bundle is exporter.export_bundle
    assert module.reset_formatters_excel is exporter.reset_formatters_excel


def test_optional_import_failures_are_tolerated(monkeypatch, load_trend_analysis):
    real_import_module = importlib.import_module
    attempts: list[str] = []

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        attempts.append(name)
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("forced missing optional module")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop("trend_analysis.data", None)
    sys.modules.pop("trend_analysis.export", None)

    module = load_trend_analysis()

    assert module is sys.modules["trend_analysis"]
    assert "trend_analysis.data" in attempts
    assert "trend_analysis.export" in attempts


def test_lazy_loader_with_stubbed_module(load_trend_analysis):
    stub = ModuleType("trend_analysis.engine")
    module = load_trend_analysis(preloaded={"trend_analysis.engine": stub})
    module.__dict__.pop("engine", None)
    assert module.engine is stub
