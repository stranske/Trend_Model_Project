from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def trend_package():
    sys.modules.pop("trend_analysis", None)
    module = importlib.import_module("trend_analysis")
    assert module.__file__ and module.__file__.endswith("trend_analysis/__init__.py")
    try:
        yield module
    finally:
        sys.modules.pop("trend_analysis", None)
        importlib.import_module("trend_analysis")


def test_lazy_attribute_import(monkeypatch, trend_package):
    dummy = types.ModuleType("trend_analysis._dummy_module")
    monkeypatch.setitem(sys.modules, "trend_analysis._dummy_module", dummy)
    monkeypatch.setitem(
        trend_package._LAZY_SUBMODULES,  # type: ignore[attr-defined]
        "_dummy",
        "trend_analysis._dummy_module",
    )
    trend_package.__dict__.pop("_dummy", None)

    loaded = trend_package.__getattr__("_dummy")

    assert loaded is dummy
    assert trend_package.__dict__["_dummy"] is dummy
    assert trend_package._dummy is dummy


def test_lazy_attribute_missing(trend_package):
    with pytest.raises(AttributeError):
        trend_package.__getattr__("does_not_exist")


def test_version_fallback(monkeypatch, trend_package):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )
    reloaded = importlib.reload(trend_package)
    assert reloaded.__version__ == "0.1.0-dev"
