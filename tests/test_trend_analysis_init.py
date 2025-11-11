"""Tests for the ``trend_analysis`` package bootstrap module."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


def _reload_trend_analysis() -> ModuleType:
    """Reload ``trend_analysis`` to ensure a clean module state for each test."""

    module = importlib.import_module("trend_analysis")
    return importlib.reload(module)


def test_patch_guard_recovers_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dataclass guard should recreate missing module entries transparently."""

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

    placeholder: ModuleType | None = None

    with monkeypatch.context() as patcher:
        patcher.setattr(dataclasses, "_is_type", stub_is_type, raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        module = importlib.reload(module)

        missing_name = "tests.fake_dataclass_module"
        dummy_cls = type("Dummy", (), {})
        dummy_cls.__module__ = missing_name
        sys.modules.pop(missing_name, None)

        try:
            result = module._SAFE_IS_TYPE(
                object, dummy_cls, None, None, lambda *_: True
            )
            placeholder = sys.modules.get(missing_name)
        finally:
            sys.modules.pop(missing_name, None)

    assert result is True
    assert isinstance(placeholder, ModuleType)
    assert placeholder.__package__ == "tests"
    importlib.reload(module)


def test_patch_guard_idempotent() -> None:
    """Running the guard twice should keep the patched helper stable."""

    module = _reload_trend_analysis()
    safe_before = module._SAFE_IS_TYPE
    module._patch_dataclasses_module_guard()
    assert module._SAFE_IS_TYPE is safe_before


def test_patch_guard_handles_absent_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """If dataclasses lacks ``_is_type`` the guard should return early."""

    module = _reload_trend_analysis()
    safe_before = module._SAFE_IS_TYPE

    with monkeypatch.context() as patcher:
        patcher.delattr(dataclasses, "_is_type", raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        module = importlib.reload(module)

    assert module._SAFE_IS_TYPE is safe_before
    importlib.reload(module)


def test_spec_proxy_re_registers_module() -> None:
    """Accessing ``__spec__.name`` should restore the module registration."""

    module = _reload_trend_analysis()
    original = sys.modules["trend_analysis"]
    sys.modules["trend_analysis"] = ModuleType("trend_analysis")

    try:
        assert module.__spec__.name == "trend_analysis"
    finally:
        sys.modules["trend_analysis"] = original

    assert sys.modules["trend_analysis"] is original


def test_lazy_attribute_imports_module() -> None:
    """Lazy attributes should import their target module on first access."""

    module = _reload_trend_analysis()
    module.__dict__.pop("presets", None)
    sys.modules.pop("trend_analysis.presets", None)

    presets = module.presets
    assert presets is sys.modules["trend_analysis.presets"]
    assert module.presets is presets


def test_safe_is_type_requires_module_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The patched helper should bubble up errors when no module name exists."""

    module = _reload_trend_analysis()

    def always_fail(*_: Any) -> bool:
        raise AttributeError("no module available")

    with monkeypatch.context() as patcher:
        patcher.setattr(dataclasses, "_is_type", always_fail, raising=False)
        patcher.delattr(dataclasses, "_trend_model_patched", raising=False)
        module = importlib.reload(module)

        nameless = type("Nameless", (), {})
        nameless.__module__ = ""

        with pytest.raises(AttributeError, match="no module available"):
            module._SAFE_IS_TYPE(object, nameless, None, None, lambda *_: True)

    importlib.reload(module)


def test_version_uses_metadata_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package metadata should take precedence when available."""

    module = _reload_trend_analysis()

    def fake_version(_: str) -> str:
        return "9.9.9"

    with monkeypatch.context() as patcher:
        patcher.setattr(importlib.metadata, "version", fake_version)
        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "9.9.9"

    importlib.reload(module)


def test_version_falls_back_when_distribution_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the distribution metadata is absent we should expose the dev fallback."""

    module = _reload_trend_analysis()

    def raise_missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    with monkeypatch.context() as patcher:
        patcher.setattr(importlib.metadata, "version", raise_missing)
        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "0.1.0-dev"

    importlib.reload(module)


def test_eager_import_skips_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eager imports should ignore optional modules that raise ImportError."""

    module = _reload_trend_analysis()
    original_import_module = importlib.import_module

    def selective_import(name: str, package: str | None = None) -> ModuleType:
        if name == "trend_analysis.metrics":
            raise ImportError("metrics optional dependency missing")
        return original_import_module(name, package)

    with monkeypatch.context() as patcher:
        patcher.setattr(importlib, "import_module", selective_import)
        module.__dict__.pop("metrics", None)
        sys.modules.pop("trend_analysis.metrics", None)
        module = importlib.reload(module)

    assert "metrics" not in module.__dict__
    importlib.reload(module)
