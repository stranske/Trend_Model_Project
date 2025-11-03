"""Tests for the package init module to ensure lazy loading and exports work."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _drop_module() -> None:
    sys.modules.pop("trend_analysis", None)


@pytest.fixture(autouse=True)
def reload_trend_analysis_after_test():
    """Reload the trend_analysis package after each test to reset globals."""

    importlib.import_module("trend_analysis")
    yield
    _drop_module()
    importlib.import_module("trend_analysis")


def _reload_with(
    monkeypatch, *, import_module=None, metadata_version=None
) -> ModuleType:
    """Reload trend_analysis with optional patches and return the module."""

    _drop_module()

    if import_module is not None:
        monkeypatch.setattr(importlib, "import_module", import_module)

    if metadata_version is not None:
        monkeypatch.setattr(importlib.metadata, "version", metadata_version)

    return importlib.import_module("trend_analysis")


def test_getattr_imports_lazy_module(monkeypatch):
    module = importlib.import_module("trend_analysis")
    module.__dict__.pop("cli", None)

    imported = module.__getattr__("cli")

    assert imported is importlib.import_module("trend_analysis.cli")
    assert module.cli is imported


def test_getattr_missing_attribute():
    module = importlib.import_module("trend_analysis")

    with pytest.raises(AttributeError):
        module.__getattr__("not_a_real_attribute")


def test_eager_import_skips_missing_module(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "trend_analysis.data":
            raise ImportError("data optional dependency unavailable")
        return real_import(name, package=package)

    module = _reload_with(monkeypatch, import_module=fake_import)

    assert "data" not in module.__dict__
    assert "export" in module.__dict__  # unaffected eager import


def test_conditional_exports_when_submodules_available():
    module = importlib.import_module("trend_analysis")

    assert "load_csv" in module.__all__
    assert callable(module.load_csv)
    assert callable(module.identify_risk_free_fund)


def test_export_guard_when_module_missing(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "trend_analysis.export":
            raise ImportError("export module unavailable")
        return real_import(name, package=package)

    module = _reload_with(monkeypatch, import_module=fake_import)

    assert "export" not in module.__dict__
    assert not hasattr(module, "export_to_csv")
    assert not hasattr(module, "export_to_json")
    assert not hasattr(module, "export_to_excel")


def test_version_from_metadata(monkeypatch):
    module = _reload_with(monkeypatch, metadata_version=lambda _: "1.2.3")

    assert module.__version__ == "1.2.3"


def test_version_fallback_when_package_missing(monkeypatch):
    def raise_missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    module = _reload_with(monkeypatch, metadata_version=raise_missing)

    assert module.__version__ == "0.1.0-dev"
