"""Tests for the ``trend_analysis`` package initialiser."""

from __future__ import annotations

import importlib
import sys
import types

import dataclasses
import importlib.metadata
import pytest
from pytest import MonkeyPatch


@pytest.fixture(name="trend_analysis_module")
def _trend_analysis_module_fixture():
    module = importlib.import_module("trend_analysis")
    module = importlib.reload(module)
    yield module
    importlib.reload(module)


def test_dataclasses_guard_recovers_missing_module(monkeypatch, trend_analysis_module):
    guard = trend_analysis_module._patch_dataclasses_module_guard

    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)

    call_count = {"calls": 0}

    def flaky_is_type(annotation, cls, a_module, a_type, predicate):
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise AttributeError("module missing")
        return True

    monkeypatch.setattr(dataclasses, "_is_type", flaky_is_type, raising=False)

    missing_name = "tests.fake_dataclass_module"
    monkeypatch.delitem(sys.modules, missing_name, raising=False)

    guard()

    dummy_cls = types.new_class("Dummy", (), {}, lambda ns: ns.update({"__module__": missing_name}))

    assert dataclasses._is_type(None, dummy_cls, None, None, None) is True
    assert missing_name in sys.modules
    restored = sys.modules[missing_name]
    assert isinstance(restored, types.ModuleType)
    assert restored.__package__ == "tests"
    assert call_count["calls"] >= 2

    sys.modules.pop(missing_name, None)


def test_spec_proxy_name_re_registers_module(monkeypatch, trend_analysis_module):
    spec = trend_analysis_module._ORIGINAL_SPEC
    proxy = trend_analysis_module._SpecProxy(spec)

    monkeypatch.delitem(sys.modules, trend_analysis_module.__name__, raising=False)

    assert proxy.name == spec.name
    assert sys.modules[trend_analysis_module.__name__] is trend_analysis_module


def test_lazy_getattr_imports_and_caches(monkeypatch, trend_analysis_module):
    sentinel = types.ModuleType("trend_analysis._lazy_test")
    target_name = "_lazy_test_alias"

    monkeypatch.setitem(trend_analysis_module._LAZY_SUBMODULES, target_name, sentinel.__name__)
    monkeypatch.delitem(trend_analysis_module.__dict__, target_name, raising=False)
    monkeypatch.setitem(sys.modules, sentinel.__name__, sentinel)

    resolved = trend_analysis_module.__getattr__(target_name)

    assert resolved is sentinel
    assert trend_analysis_module.__dict__[target_name] is sentinel


def test_lazy_getattr_unknown_attribute_raises(trend_analysis_module):
    with pytest.raises(AttributeError):
        trend_analysis_module.__getattr__("does_not_exist")


def test_dataclasses_guard_raises_when_module_unknown(monkeypatch, trend_analysis_module):
    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)

    def always_missing(annotation, cls, a_module, a_type, predicate):
        raise AttributeError("missing module")

    monkeypatch.setattr(dataclasses, "_is_type", always_missing, raising=False)

    guard = trend_analysis_module._patch_dataclasses_module_guard
    guard()

    dummy_cls = types.new_class("DummyNoModule")
    dummy_cls.__module__ = None

    with pytest.raises(AttributeError):
        dataclasses._is_type(None, dummy_cls, None, None, None)


def test_eager_import_skips_missing_modules():
    monkey = MonkeyPatch()
    try:
        monkey.delitem(sys.modules, "trend_analysis", raising=False)

        importlib_module = importlib
        real_import = importlib_module.import_module

        def fake_import(name, package=None):
            if name == "trend_analysis.metrics":
                raise ImportError("simulated missing optional dependency")
            return real_import(name, package)

        monkey.setattr(importlib_module, "import_module", fake_import)
        module = importlib_module.import_module("trend_analysis")

        assert "metrics" not in module.__dict__
    finally:
        monkey.undo()
        importlib.reload(importlib.import_module("trend_analysis"))


def test_spec_proxy_block_assigns_proxy(trend_analysis_module):
    spec = types.SimpleNamespace(name="trend_analysis")
    original = trend_analysis_module.__dict__.get("_ORIGINAL_SPEC")
    trend_analysis_module._ORIGINAL_SPEC = spec

    exec(
        "if _ORIGINAL_SPEC is not None:\n"
        "    globals()['__spec__'] = _SpecProxy(_ORIGINAL_SPEC)\n"
        "_ensure_registered()",
        trend_analysis_module.__dict__,
    )

    assert isinstance(trend_analysis_module.__spec__, trend_analysis_module._SpecProxy)
    trend_analysis_module._ORIGINAL_SPEC = original


def test_conditional_exports_block_runs(trend_analysis_module):
    exec(
        "if 'data' in globals():\n"
        "    from .data import identify_risk_free_fund, load_csv\n"
        "if 'export' in globals():\n"
        "    from .export import (\n"
        "        combined_summary_frame,\n"
        "        combined_summary_result,\n"
        "        export_bundle,\n"
        "        export_data,\n"
        "        export_multi_period_metrics,\n"
        "        export_phase1_multi_metrics,\n"
        "        export_phase1_workbook,\n"
        "        export_to_csv,\n"
        "        export_to_excel,\n"
        "        export_to_json,\n"
        "        export_to_txt,\n"
        "        flat_frames_from_results,\n"
        "        make_summary_formatter,\n"
        "        metrics_from_result,\n"
        "        phase1_workbook_data,\n"
        "        register_formatter_excel,\n"
        "        reset_formatters_excel,\n"
        "    )",
        trend_analysis_module.__dict__,
    )

    assert trend_analysis_module.load_csv is trend_analysis_module.data.load_csv
    assert trend_analysis_module.export_to_json is trend_analysis_module.export.export_to_json


def test_version_block_falls_back():
    monkey = MonkeyPatch()
    try:
        monkey.delitem(sys.modules, "trend_analysis", raising=False)

        def missing_version(package_name):
            raise importlib.metadata.PackageNotFoundError

        monkey.setattr(importlib.metadata, "version", missing_version)
        module = importlib.import_module("trend_analysis")

        assert module.__version__ == "0.1.0-dev"
    finally:
        monkey.undo()
        importlib.reload(importlib.import_module("trend_analysis"))
