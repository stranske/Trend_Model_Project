"""Additional coverage for ``trend_analysis.__init__`` guard rails."""

from __future__ import annotations

import sys
from types import ModuleType

import dataclasses
import pytest

import trend_analysis


@pytest.fixture()
def restore_dataclasses_is_type():
    original = dataclasses._is_type
    original_flag = getattr(dataclasses, "_trend_model_patched", None)
    try:
        yield
    finally:
        dataclasses._is_type = original
        if original_flag is None:
            if hasattr(dataclasses, "_trend_model_patched"):
                delattr(dataclasses, "_trend_model_patched")
        else:
            dataclasses._trend_model_patched = original_flag  # type: ignore[attr-defined]
        trend_analysis._patch_dataclasses_module_guard()


def _reset_guard() -> None:
    if hasattr(dataclasses, "_trend_model_patched"):
        delattr(dataclasses, "_trend_model_patched")


def test_dataclasses_guard_reimports_missing_module(monkeypatch, restore_dataclasses_is_type):
    module_name = "tests.fake_missing_module"
    created = ModuleType(module_name)

    call_count: dict[str, int] = {"calls": 0}

    def fake_original(annotation, cls, a_module, a_type, predicate):  # type: ignore[no-untyped-def]
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise AttributeError("module missing")
        assert sys.modules[module_name] is created
        return True

    def fake_import(name: str) -> ModuleType:
        assert name == module_name
        sys.modules[name] = created
        return created

    dataclasses._is_type = fake_original  # type: ignore[assignment]
    monkeypatch.setattr(trend_analysis.importlib, "import_module", fake_import)

    _reset_guard()
    trend_analysis._patch_dataclasses_module_guard()

    class Dummy:
        __module__ = module_name

    sys.modules.pop(module_name, None)
    assert dataclasses._is_type(None, Dummy, None, None, None) is True
    assert call_count["calls"] == 2


def test_dataclasses_guard_fallback_creates_placeholder(monkeypatch, restore_dataclasses_is_type):
    module_name = "tests.nonexistent_placeholder"

    call_count = {"calls": 0}

    def fake_original(annotation, cls, a_module, a_type, predicate):  # type: ignore[no-untyped-def]
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise AttributeError("module missing")
        return False

    def failing_import(name: str) -> ModuleType:
        assert name == module_name
        raise ImportError("boom")

    dataclasses._is_type = fake_original  # type: ignore[assignment]
    monkeypatch.setattr(trend_analysis.importlib, "import_module", failing_import)

    _reset_guard()
    trend_analysis._patch_dataclasses_module_guard()

    class Dummy:
        __module__ = module_name

    sys.modules.pop(module_name, None)
    result = dataclasses._is_type(None, Dummy, None, None, None)
    assert result is False
    placeholder = sys.modules[module_name]
    assert isinstance(placeholder, ModuleType)
    assert placeholder.__package__ == "tests"


def test_dataclasses_guard_propagates_without_module(restore_dataclasses_is_type):
    call_count = {"calls": 0}

    def fake_original(annotation, cls, a_module, a_type, predicate):  # type: ignore[no-untyped-def]
        call_count["calls"] += 1
        raise AttributeError("missing module")

    dataclasses._is_type = fake_original  # type: ignore[assignment]
    _reset_guard()
    trend_analysis._patch_dataclasses_module_guard()

    class Dummy:
        __module__ = ""

    with pytest.raises(AttributeError):
        dataclasses._is_type(None, Dummy, None, None, None)
    assert call_count["calls"] == 1


def test_dataclasses_guard_uses_existing_module(restore_dataclasses_is_type):
    module_name = "tests.preloaded_module"
    sentinel = ModuleType(module_name)
    call_count = {"calls": 0}

    def fake_original(annotation, cls, a_module, a_type, predicate):  # type: ignore[no-untyped-def]
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise AttributeError("module missing")
        assert sys.modules[module_name] is sentinel
        return True

    sys.modules[module_name] = sentinel
    dataclasses._is_type = fake_original  # type: ignore[assignment]
    _reset_guard()
    trend_analysis._patch_dataclasses_module_guard()

    class Dummy:
        __module__ = module_name

    assert dataclasses._is_type(None, Dummy, None, None, None) is True
    assert call_count["calls"] == 2
    sys.modules.pop(module_name, None)


def test_spec_proxy_restores_module_registration(monkeypatch):
    sentinel_module = ModuleType("trend_analysis")
    original = sys.modules["trend_analysis"]
    monkeypatch.setitem(sys.modules, "trend_analysis", sentinel_module)

    class DummySpec:
        name = "trend_analysis"

    proxy = trend_analysis._SpecProxy(DummySpec())
    with pytest.raises(AttributeError):
        _ = proxy.missing

    name = proxy.name
    assert name == getattr(proxy._spec, "name", None)
    assert sys.modules["trend_analysis"] is trend_analysis
