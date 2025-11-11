"""Soft-coverage smoke tests for :mod:`trend_analysis.__init__`.

These tests focus on execution paths that matter for importing the public
package. They are intentionally lightweight so the soft coverage suite can
exercise defensive behaviours without the cost of the full regression tests.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


def _reload_trend_analysis() -> ModuleType:
    module = importlib.import_module("trend_analysis")
    return importlib.reload(module)


def test_dataclass_guard_recovers_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_trend_analysis()
    call_counter = {"count": 0}

    def stub_is_type(
        annotation: Any,
        cls: type[Any],
        a_module: Any,
        a_type: Any,
        predicate: Any,
    ) -> bool:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise AttributeError("module missing during dataclass resolution")
        return True

    missing_name = "tests.soft_guard_module"

    with monkeypatch.context() as patcher:
        patcher.setattr(dataclasses, "_is_type", stub_is_type, raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        reloaded = importlib.reload(module)

        dummy_cls = type("Dummy", (), {})
        dummy_cls.__module__ = missing_name
        sys.modules.pop(missing_name, None)

        assert reloaded._SAFE_IS_TYPE(object, dummy_cls, None, None, lambda *_: True)
        assert missing_name in sys.modules

    sys.modules.pop(missing_name, None)
    importlib.reload(module)


def test_spec_proxy_re_registers_module() -> None:
    module = _reload_trend_analysis()
    original = sys.modules["trend_analysis"]
    sys.modules["trend_analysis"] = ModuleType("trend_analysis")

    try:
        assert module.__spec__.name == "trend_analysis"
    finally:
        sys.modules["trend_analysis"] = original

    assert sys.modules["trend_analysis"] is original


def test_lazy_attribute_imports_on_demand() -> None:
    module = _reload_trend_analysis()
    module.__dict__.pop("presets", None)
    sys.modules.pop("trend_analysis.presets", None)

    presets = module.presets
    assert presets is sys.modules["trend_analysis.presets"]
    assert module.presets is presets
    importlib.reload(module)


def test_version_fallback_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_trend_analysis()

    def raise_missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    with monkeypatch.context() as patcher:
        patcher.setattr(importlib.metadata, "version", raise_missing)
        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "0.1.0-dev"

    importlib.reload(module)


def test_guard_handles_absent_original(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_trend_analysis()

    with monkeypatch.context() as patcher:
        patcher.delattr(dataclasses, "_is_type", raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        reloaded = importlib.reload(module)

    assert hasattr(reloaded, "_SAFE_IS_TYPE")
    importlib.reload(module)


def test_guard_reuses_existing_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_trend_analysis()

    sentinel = object()
    with monkeypatch.context() as patcher:
        patcher.setattr(dataclasses, "_trend_model_patched", True, raising=False)
        patcher.setattr(dataclasses, "_is_type", sentinel, raising=False)
        reloaded = importlib.reload(module)

    assert reloaded._SAFE_IS_TYPE is sentinel
    importlib.reload(module)


def test_lazy_getattr_rejects_unknown() -> None:
    module = _reload_trend_analysis()
    with pytest.raises(AttributeError):
        getattr(module, "not_a_module")


def test_ensure_registered_restores_module_entry() -> None:
    module = _reload_trend_analysis()
    sys.modules["trend_analysis"] = ModuleType("trend_analysis")
    try:
        module._ensure_registered()
        assert sys.modules["trend_analysis"] is module
    finally:
        sys.modules["trend_analysis"] = module


def test_export_attributes_omitted_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_trend_analysis()
    original_import = importlib.import_module

    def selective(name: str, package: str | None = None) -> ModuleType:
        if name == "trend_analysis.export":
            raise ImportError("export optional dependency missing")
        return original_import(name, package)

    with monkeypatch.context() as patcher:
        patcher.setattr(importlib, "import_module", selective)
        module.__dict__.pop("export", None)
        sys.modules.pop("trend_analysis.export", None)
        for attr in {
            "export_to_csv",
            "export_to_json",
            "export_to_excel",
            "export_to_txt",
            "export_data",
        }:
            module.__dict__.pop(attr, None)
        reloaded = importlib.reload(module)

    assert "export" not in reloaded.__dict__
    assert "export_to_csv" not in reloaded.__dict__
    importlib.reload(module)


def test_safe_is_type_requires_module_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_trend_analysis()

    def always_fail(*_: Any) -> bool:
        raise AttributeError("no module available")

    with monkeypatch.context() as patcher:
        patcher.setattr(dataclasses, "_is_type", always_fail, raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        reloaded = importlib.reload(module)
        nameless = type("Nameless", (), {})
        nameless.__module__ = ""
        with pytest.raises(AttributeError, match="no module available"):
            reloaded._SAFE_IS_TYPE(object, nameless, None, None, lambda *_: True)

    importlib.reload(module)
