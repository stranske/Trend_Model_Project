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


def test_eager_import_skips_missing_submodule(monkeypatch):
    """Eager import should quietly skip optional modules that fail to load."""

    original_import = importlib.import_module

    # Remove cached modules so the eager import guard has to execute.
    for name in [
        "trend_analysis",
        "trend_analysis.metrics",
        "trend_analysis.config",
        "trend_analysis.data",
        "trend_analysis.pipeline",
        "trend_analysis.export",
        "trend_analysis.signals",
        "trend_analysis.backtesting",
    ]:
        sys.modules.pop(name, None)

    failures: set[str] = set()

    def fail_once(name: str, package: str | None = None):
        if name == "trend_analysis.metrics" and name not in failures:
            failures.add(name)
            raise ImportError("missing optional dependency")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_once)
    module = importlib.import_module("trend_analysis")
    assert "trend_analysis.metrics" in failures

    # Exercise optional imports missing entirely (``data`` and ``export`` guards).
    def skip_optional(name: str, package: str | None = None):
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("optional module unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", skip_optional)
    for name in [
        "trend_analysis",
        "trend_analysis.data",
        "trend_analysis.export",
    ]:
        sys.modules.pop(name, None)

    missing_optional = importlib.import_module("trend_analysis")
    assert "identify_risk_free_fund" not in missing_optional.__dict__
    assert "export_bundle" not in missing_optional.__dict__

    # Reload with the real import machinery so later tests see the genuine module.
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
        lambda package: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError()
        ),
    )
    reloaded = importlib.reload(trend_package)
    assert reloaded.__version__ == "0.1.0-dev"
