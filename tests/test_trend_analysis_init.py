"""Tests for the :mod:`trend_analysis` package initialisation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture
def fresh_trend_module() -> ModuleType:
    import trend_analysis

    return importlib.reload(trend_analysis)


def test_version_fallback(monkeypatch: pytest.MonkeyPatch, fresh_trend_module: ModuleType) -> None:
    import importlib.metadata

    def fake_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    reloaded = importlib.reload(fresh_trend_module)
    assert reloaded.__version__ == "0.1.0-dev"


def test_lazy_import(monkeypatch: pytest.MonkeyPatch, fresh_trend_module: ModuleType) -> None:
    sentinel = ModuleType("trend_analysis.selector")
    monkeypatch.setitem(sys.modules, "trend_analysis.selector", sentinel)
    reloaded = importlib.reload(fresh_trend_module)

    assert "selector" not in reloaded.__dict__
    assert reloaded.selector is sentinel


def test_optional_module_failures_suppressed(monkeypatch: pytest.MonkeyPatch, fresh_trend_module: ModuleType) -> None:
    import trend_analysis

    original_import = importlib.import_module

    def stub_import(name: str, package: str | None = None):  # type: ignore[override]
        if name == "trend_analysis.export":
            raise ImportError("optional module missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", stub_import)
    fresh_trend_module.__dict__.pop("export", None)
    monkeypatch.delitem(sys.modules, "trend_analysis.export", raising=False)
    reloaded = importlib.reload(fresh_trend_module)

    assert "export" not in reloaded.__dict__
    # Attribute access should still raise AttributeError
    with pytest.raises(AttributeError):
        getattr(reloaded, "nonexistent")

    # Restore real import for subsequent tests
    importlib.reload(trend_analysis)


def test_reexports_available(fresh_trend_module: ModuleType) -> None:
    assert callable(fresh_trend_module.load_csv)
    assert callable(fresh_trend_module.export_to_csv)


def test_data_module_missing(monkeypatch: pytest.MonkeyPatch, fresh_trend_module: ModuleType) -> None:
    original_import = importlib.import_module

    def stub_import(name: str, package: str | None = None):  # type: ignore[override]
        if name == "trend_analysis.data":
            raise ImportError("optional data module missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", stub_import)
    fresh_trend_module.__dict__.pop("data", None)
    fresh_trend_module.__dict__.pop("load_csv", None)
    fresh_trend_module.__dict__.pop("identify_risk_free_fund", None)
    monkeypatch.delitem(sys.modules, "trend_analysis.data", raising=False)
    reloaded = importlib.reload(fresh_trend_module)

    with pytest.raises(AttributeError):
        getattr(reloaded, "load_csv")
