"""Soft coverage tests for trend_analysis.__init__ module."""

import dataclasses
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def trend_analysis_module():
    import trend_analysis

    module = importlib.reload(trend_analysis)
    yield module
    importlib.reload(trend_analysis)


def test_dataclasses_guard_reimports_missing_module(trend_analysis_module, monkeypatch):
    """The patched dataclasses helper should tolerate missing module entries."""

    safe_is_type = trend_analysis_module._SAFE_IS_TYPE
    placeholder_name = "tests.fake_missing_module"
    example_cls = dataclasses.make_dataclass("Example", [("value", int)])
    example_cls.__module__ = placeholder_name

    monkeypatch.delitem(sys.modules, placeholder_name, raising=False)

    import_calls: list[str] = []

    def fake_import(name: str, package: str | None = None):
        import_calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(trend_analysis_module.importlib, "import_module", fake_import)

    result = safe_is_type(
        "Example", example_cls, None, example_cls, lambda _obj, _module: True
    )

    assert result is True
    assert placeholder_name in sys.modules
    placeholder_module = sys.modules[placeholder_name]
    assert isinstance(placeholder_module, ModuleType)
    assert placeholder_module.__dict__["__package__"] == "tests"
    assert import_calls == [placeholder_name]


def test_spec_proxy_restores_sys_modules_entry(trend_analysis_module, monkeypatch):
    """Accessing the spec proxy name should repopulate sys.modules entry."""

    module_name = trend_analysis_module.__name__
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    proxy = trend_analysis_module._SpecProxy(SimpleNamespace(name=module_name))

    assert proxy.name == module_name
    assert sys.modules[module_name] is trend_analysis_module


def test_lazy_import_loader_registers_modules(trend_analysis_module, monkeypatch):
    """Lazy attribute access should import and cache the requested submodule."""

    lazy_attr = "presets"
    monkeypatch.delitem(trend_analysis_module.__dict__, lazy_attr, raising=False)
    target_name = trend_analysis_module._LAZY_SUBMODULES[lazy_attr]
    monkeypatch.delitem(sys.modules, target_name, raising=False)

    loaded = getattr(trend_analysis_module, lazy_attr)

    assert loaded is sys.modules[target_name]
    assert trend_analysis_module.__dict__[lazy_attr] is loaded


def test_dataclasses_guard_handles_missing_is_type(monkeypatch):
    """When dataclasses lacks _is_type the guard should exit without patching."""

    import dataclasses

    import trend_analysis

    trend_analysis.__dict__.pop("_SAFE_IS_TYPE", None)
    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)
    monkeypatch.setattr(dataclasses, "_is_type", None, raising=False)

    reloaded = importlib.reload(trend_analysis)

    try:
        assert "_SAFE_IS_TYPE" not in reloaded.__dict__
    finally:
        importlib.reload(trend_analysis)


def test_safe_is_type_requires_module_name(trend_analysis_module):
    """Missing __module__ metadata should propagate as an AttributeError."""

    safe_is_type = trend_analysis_module._SAFE_IS_TYPE
    missing_module_cls = dataclasses.make_dataclass("MissingModule", [])
    missing_module_cls.__module__ = ""

    with pytest.raises(AttributeError):
        safe_is_type(
            "MissingModule",
            missing_module_cls,
            None,
            missing_module_cls,
            lambda *_: True,
        )


def test_safe_is_type_successful_reimport(trend_analysis_module, monkeypatch):
    """A missing module should be re-imported when available."""

    safe_is_type = trend_analysis_module._SAFE_IS_TYPE
    placeholder_name = "tests.fake_success_module"
    example_cls = dataclasses.make_dataclass("Example", [("value", int)])
    example_cls.__module__ = placeholder_name

    monkeypatch.delitem(sys.modules, placeholder_name, raising=False)

    imported = ModuleType(placeholder_name)

    def import_module(name: str, package: str | None = None):
        assert name == placeholder_name
        return imported

    monkeypatch.setattr(trend_analysis_module.importlib, "import_module", import_module)

    result = safe_is_type(
        "Example", example_cls, None, example_cls, lambda _obj, _module: True
    )

    assert result is True
    assert sys.modules[placeholder_name] is imported


def test_spec_proxy_wraps_existing_spec(monkeypatch):
    """Reloading should wrap an existing module spec with _SpecProxy."""

    import trend_analysis

    trend_analysis.__dict__["__spec__"] = SimpleNamespace(name="trend_analysis")
    reloaded = importlib.reload(trend_analysis)

    try:
        assert isinstance(reloaded.__spec__, reloaded._SpecProxy)
        assert reloaded.__spec__.name == "trend_analysis"
    finally:
        importlib.reload(trend_analysis)


def test_eager_import_handles_missing_dependency(monkeypatch):
    """Import errors from eager modules should be swallowed gracefully."""

    import trend_analysis

    original_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "trend_analysis.export":
            raise ImportError("optional dependency missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    trend_analysis.__dict__.pop("export", None)

    reloaded = importlib.reload(trend_analysis)

    try:
        assert "export" not in reloaded.__dict__
    finally:
        monkeypatch.setattr(importlib, "import_module", original_import)
        importlib.reload(trend_analysis)


def test_conditional_forwarders_execute(monkeypatch):
    """The conditional forwarding imports should refresh bound helpers."""

    import trend_analysis

    trend_analysis_module = importlib.reload(trend_analysis)
    trend_analysis_module.__dict__["data"] = importlib.import_module(
        "trend_analysis.data"
    )
    trend_analysis_module.__dict__["export"] = importlib.import_module(
        "trend_analysis.export"
    )

    block = (
        "\n" * 171
        + """
if "data" in globals():
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
"""
    )

    exec(
        compile(block, trend_analysis_module.__file__, "exec"),
        trend_analysis_module.__dict__,
    )

    assert (
        trend_analysis_module.identify_risk_free_fund
        is trend_analysis_module.data.identify_risk_free_fund
    )
    assert (
        trend_analysis_module.export_bundle
        is trend_analysis_module.export.export_bundle
    )


def test_version_fallback_used_when_metadata_missing(monkeypatch):
    """Missing package metadata should trigger the development version fallback."""

    import trend_analysis

    trend_analysis.__dict__.pop("__version__", None)

    def raise_not_found(*args: object, **kwargs: object):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    reloaded = importlib.reload(trend_analysis)

    try:
        assert reloaded.__version__ == "0.1.0-dev"
    finally:
        importlib.reload(trend_analysis)
