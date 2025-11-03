"""Focused coverage for :mod:`trend_analysis` package initialisation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture()
def fresh_trend_analysis():
    """Provide a freshly reloaded copy of the package for isolated tests."""

    if "trend_analysis" in sys.modules:
        module = importlib.reload(sys.modules["trend_analysis"])
    else:
        module = importlib.import_module("trend_analysis")
    try:
        yield module
    finally:
        importlib.reload(module)


def test_lazy_getattr_imports_module(monkeypatch, fresh_trend_analysis):
    """Accessing lazy attributes should import the declared module."""

    package = fresh_trend_analysis
    sentinel = ModuleType("trend_analysis.selector")
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "trend_analysis.selector":
            return sentinel
        return original_import(name, package)

    monkeypatch.setattr(
        package, "_LAZY_SUBMODULES", {"selector": "trend_analysis.selector"}
    )
    monkeypatch.delattr(package, "selector", raising=False)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    loaded = package.selector
    assert loaded is sentinel
    assert package.selector is sentinel  # cached for subsequent lookups


def test_missing_attribute_raises_attribute_error(fresh_trend_analysis):
    package = fresh_trend_analysis
    with pytest.raises(AttributeError):
        getattr(package, "does_not_exist")


def test_version_fallback_when_package_metadata_missing(monkeypatch):
    """Reloading should populate the development fallback version."""

    from importlib.metadata import PackageNotFoundError

    # Ensure a clean import path for the package under test
    sys.modules.pop("trend_analysis", None)

    import trend_analysis as ta

    def raise_missing(name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_missing)
    reloaded = importlib.reload(ta)
    assert reloaded.__version__ == "0.1.0-dev"

    # Restore the real version loader for subsequent imports
    importlib.reload(reloaded)


def test_public_exports_include_load_csv(fresh_trend_analysis):
    package = fresh_trend_analysis
    data_module = importlib.import_module("trend_analysis.data")

    assert package.load_csv is data_module.load_csv
    assert package.identify_risk_free_fund is data_module.identify_risk_free_fund


def test_optional_import_failures_are_handled(monkeypatch):
    """Missing optional dependencies should not break package import."""

    sys.modules.pop("trend_analysis", None)

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("optional dependency missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    import trend_analysis as ta

    reloaded = importlib.reload(ta)

    assert not hasattr(reloaded, "load_csv")
    assert not hasattr(reloaded, "export_data")

    importlib.reload(reloaded)


def test_version_metadata_success_path(monkeypatch):
    """Ensure the package records the resolved distribution version."""

    sys.modules.pop("trend_analysis", None)
    import trend_analysis as ta

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    reloaded = importlib.reload(ta)
    assert reloaded.__version__ == "9.9.9"

    importlib.reload(reloaded)
