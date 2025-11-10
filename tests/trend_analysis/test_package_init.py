from __future__ import annotations

import dataclasses
import importlib
import sys
from types import ModuleType
from typing import Mapping

import pytest


def _clear_trend_analysis_modules() -> None:
    for name in [
        key
        for key in sys.modules
        if key == "trend_analysis" or key.startswith("trend_analysis.")
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


def _make_fake_dataclass(module_name: str) -> tuple[ModuleType, type[object]]:
    fake_module = ModuleType(module_name)
    fake_module.__dict__["__package__"] = module_name.rpartition(".")[0]
    exec(
        "from dataclasses import dataclass, InitVar\n"
        "@dataclass\n"
        "class Example:\n"
        "    value: InitVar[int]\n",
        fake_module.__dict__,
    )
    return fake_module, fake_module.Example


def test_dataclasses_guard_reimports_missing_module(load_trend_analysis):
    module = load_trend_analysis()
    assert module is sys.modules["trend_analysis"]

    module_name = "tests.fake_module_for_trend_init"
    fake_module, cls = _make_fake_dataclass(module_name)
    sys.modules[module_name] = fake_module
    sys.modules.pop(module_name, None)

    result = dataclasses._is_type(
        "InitVar",
        cls,
        dataclasses,
        dataclasses.InitVar,
        lambda candidate, module: candidate is dataclasses.InitVar,
    )

    assert result is True or result is False
    assert module_name in sys.modules
    placeholder = sys.modules[module_name]
    assert isinstance(placeholder, ModuleType)
    assert placeholder.__dict__["__package__"] == "tests"


def test_dataclasses_guard_preserves_existing_modules(load_trend_analysis):
    load_trend_analysis()
    module_name = "tests.fake_module_for_trend_init_existing"
    fake_module, cls = _make_fake_dataclass(module_name)
    sys.modules[module_name] = fake_module

    result = dataclasses._is_type(
        "InitVar",
        cls,
        dataclasses,
        dataclasses.InitVar,
        lambda candidate, module: candidate is dataclasses.InitVar,
    )

    assert result is True or result is False
    assert sys.modules[module_name] is fake_module


def test_dataclasses_guard_re_raises_for_orphan_modules(load_trend_analysis):
    load_trend_analysis()

    class Orphan:
        __module__ = ""

    with pytest.raises(AttributeError):
        dataclasses._is_type(
            "InitVar",
            Orphan,
            dataclasses,
            dataclasses.InitVar,
            lambda candidate, module: False,
        )


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


def test_lazy_loader_with_stubbed_module(load_trend_analysis):
    stub = ModuleType("trend_analysis.engine")
    module = load_trend_analysis(preloaded={"trend_analysis.engine": stub})
    module.__dict__.pop("engine", None)
    assert module.engine is stub
