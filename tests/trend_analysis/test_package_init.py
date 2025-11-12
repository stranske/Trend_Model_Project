from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
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
    sys.modules.pop("tests.fake_module_for_trend_init_retry", None)


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


def test_patch_skips_when_original_is_missing(monkeypatch, load_trend_analysis):
    module = load_trend_analysis()
    original = dataclasses._is_type

    previous_safe = module._SAFE_IS_TYPE  # type: ignore[attr-defined]

    monkeypatch.setattr(dataclasses, "_is_type", None, raising=False)
    monkeypatch.setattr(dataclasses, "_trend_model_patched", False, raising=False)

    module._patch_dataclasses_module_guard()

    assert module._SAFE_IS_TYPE is previous_safe  # type: ignore[attr-defined]
    assert getattr(dataclasses, "_is_type") is None
    assert not getattr(dataclasses, "_trend_model_patched", False)

    dataclasses._is_type = original  # type: ignore[attr-defined]
    dataclasses._trend_model_patched = False  # type: ignore[attr-defined]
    module._patch_dataclasses_module_guard()


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

    assert isinstance(result, bool)
    assert sys.modules[module_name] is fake_module


def test_patch_handles_attribute_error_and_reimports(monkeypatch, load_trend_analysis):
    module = load_trend_analysis()
    module_name = "tests.fake_module_for_trend_init_retry"
    calls: list[str] = []

    def stub(
        annotation: object,
        cls: type[object],
        a_module: object,
        a_type: object,
        predicate: object,
    ) -> object:
        calls.append(cls.__module__)
        if cls.__module__ == module_name and module_name not in sys.modules:
            raise AttributeError("missing module")
        if callable(predicate):
            return predicate(a_type, a_module)  # type: ignore[no-any-return]
        return predicate

    monkeypatch.setattr(dataclasses, "_is_type", stub, raising=False)
    monkeypatch.setattr(dataclasses, "_trend_model_patched", False, raising=False)

    module._patch_dataclasses_module_guard()
    safe = dataclasses._is_type

    fake_module, cls = _make_fake_dataclass(module_name)
    sys.modules[module_name] = fake_module
    sys.modules.pop(module_name, None)

    result = safe(
        "InitVar",
        cls,
        dataclasses,
        dataclasses.InitVar,
        lambda candidate, module: candidate is dataclasses.InitVar,
    )

    assert calls.count(module_name) >= 2
    assert result is True
    placeholder = sys.modules[module_name]
    assert isinstance(placeholder, ModuleType)
    assert placeholder.__dict__["__package__"] == "tests"

    dataclasses._is_type = module._SAFE_IS_TYPE  # type: ignore[attr-defined]
    dataclasses._trend_model_patched = True  # type: ignore[attr-defined]


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


def test_patch_noop_when_already_wrapped(load_trend_analysis):
    module = load_trend_analysis()
    initial = dataclasses._is_type
    module._patch_dataclasses_module_guard()
    assert dataclasses._is_type is initial
    assert module._SAFE_IS_TYPE is initial  # type: ignore[attr-defined]


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
