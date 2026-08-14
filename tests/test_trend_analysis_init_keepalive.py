"""Additional coverage for ``trend_analysis.__init__`` guard rails."""

from __future__ import annotations

from types import ModuleType

import pytest

import trend_analysis


def test_spec_proxy_restores_module_registration(monkeypatch):
    sentinel_module = ModuleType("trend_analysis")
    monkeypatch.setitem(__import__("sys").modules, "trend_analysis", sentinel_module)

    class DummySpec:
        name = "trend_analysis"

    proxy = trend_analysis._SpecProxy(DummySpec())
    with pytest.raises(AttributeError):
        _ = proxy.missing

    name = proxy.name
    assert name == DummySpec().name
    assert __import__("sys").modules["trend_analysis"] is trend_analysis
