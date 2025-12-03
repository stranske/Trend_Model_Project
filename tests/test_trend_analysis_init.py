"""Coverage-focused tests for ``trend_analysis.__init__`` internals."""

import importlib
import sys
from types import ModuleType

import pytest


def test_ensure_registered_restores_module(monkeypatch):
    import trend_analysis as ta

    placeholder = ModuleType("trend_analysis")
    monkeypatch.setitem(sys.modules, "trend_analysis", placeholder)

    ta._ensure_registered()

    assert sys.modules["trend_analysis"] is ta


def test_getattr_lazy_import(monkeypatch):
    import trend_analysis as ta

    dummy_mod = ModuleType("trend_analysis.dummy_lazy")
    monkeypatch.setitem(sys.modules, "trend_analysis.dummy_lazy", dummy_mod)
    monkeypatch.setitem(ta._LAZY_SUBMODULES, "dummy_attr", "trend_analysis.dummy_lazy")
    monkeypatch.delitem(ta.__dict__, "dummy_attr", raising=False)

    result = getattr(ta, "dummy_attr")

    assert result is dummy_mod
    assert ta.dummy_attr is dummy_mod


def test_getattr_unknown_raises_attribute_error():
    import trend_analysis as ta

    with pytest.raises(AttributeError):
        getattr(ta, "non_existent_submodule")


def test_safe_is_type_handles_missing_module(monkeypatch):
    import dataclasses

    def flaky_is_type(annotation, cls, a_module, a_type, predicate):
        if cls.__module__ not in sys.modules:
            raise AttributeError("module missing")
        return True

    original_module = importlib.import_module("trend_analysis")

    monkeypatch.setattr(dataclasses, "_is_type", flaky_is_type)
    monkeypatch.setattr(dataclasses, "_trend_model_patched", False, raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis", raising=False)
    monkeypatch.delitem(sys.modules, "tests.missing_mod", raising=False)

    ta = importlib.reload(original_module)

    Dummy = type("Dummy", (), {"__module__": "tests.missing_mod"})

    assert ta._SAFE_IS_TYPE(None, Dummy, None, None, lambda _: False) is True
    assert "tests.missing_mod" in sys.modules


def test_spec_proxy_triggers_registration(monkeypatch):
    import trend_analysis as ta

    class Spec:
        name = "trend_analysis"

    proxy = ta._SpecProxy(Spec())

    monkeypatch.setitem(sys.modules, "trend_analysis", ModuleType("trend_analysis"))

    assert proxy.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is ta
