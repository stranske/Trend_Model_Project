import importlib
import importlib.metadata
import sys
from types import ModuleType
from typing import Optional

import pytest


def test_lazy_getattr_loads_and_caches_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    sentinel = ModuleType("trend_analysis.selector")
    setattr(sentinel, "flag", "sentinel")

    original_import = importlib.import_module

    def fake_import(name: str, package: Optional[str] = None) -> ModuleType:
        if name == "trend_analysis.selector":
            return sentinel
        return original_import(name, package=package)

    monkeypatch.delattr(trend_analysis, "selector", raising=False)
    monkeypatch.setattr(importlib, "import_module", fake_import)

    loaded = trend_analysis.__getattr__("selector")

    assert loaded is sentinel
    assert trend_analysis.selector is sentinel


def test_getattr_unknown_name_raises_attribute_error() -> None:
    import trend_analysis

    with pytest.raises(AttributeError):
        trend_analysis.__getattr__("does_not_exist")


def test_version_fallback_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("trend_analysis")

    with monkeypatch.context() as m:

        def fake_version(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        m.setattr(importlib.metadata, "version", fake_version)

        reloaded = importlib.reload(module)
        assert reloaded.__version__ == "0.1.0-dev"

    importlib.reload(module)


def test_top_level_reexports_expose_data_and_export_helpers() -> None:
    module = importlib.reload(importlib.import_module("trend_analysis"))

    # ``load_csv`` and ``identify_risk_free_fund`` should come from the data module.
    assert hasattr(module, "load_csv")
    assert module.load_csv is module.data.load_csv
    assert hasattr(module, "identify_risk_free_fund")
    assert module.identify_risk_free_fund is module.data.identify_risk_free_fund

    # Export helpers are re-exported when the export submodule imports succeed.
    assert hasattr(module, "export_to_csv")
    assert module.export_to_csv is module.export.export_to_csv


def test_eager_import_skips_missing_optional_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(importlib.import_module("trend_analysis"))
    sys.modules.pop("trend_analysis.signals", None)

    original_import = importlib.import_module

    def fake_import(name: str, package: Optional[str] = None) -> ModuleType:
        if name == "trend_analysis.signals":
            raise ImportError("optional dependency not available")
        return original_import(name, package=package)

    with monkeypatch.context() as context:
        context.delattr(module, "signals", raising=False)
        context.setattr(importlib, "import_module", fake_import)
        reloaded = importlib.reload(module)

        assert not hasattr(reloaded, "signals")

    importlib.reload(module)


def test_conditional_reexports_skip_when_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.reload(importlib.import_module("trend_analysis"))
    for name in ("trend_analysis.data", "trend_analysis.export"):
        sys.modules.pop(name, None)

    original_import = importlib.import_module

    def fake_import(name: str, package: Optional[str] = None) -> ModuleType:
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("optional dependency not available")
        return original_import(name, package=package)

    with monkeypatch.context() as context:
        context.delattr(module, "data", raising=False)
        context.delattr(module, "export", raising=False)
        context.delattr(module, "load_csv", raising=False)
        context.delattr(module, "identify_risk_free_fund", raising=False)
        context.delattr(module, "export_to_csv", raising=False)
        context.delattr(module, "export_data", raising=False)
        context.setattr(importlib, "import_module", fake_import)
        reloaded = importlib.reload(module)

        assert not hasattr(reloaded, "load_csv")
        assert not hasattr(reloaded, "identify_risk_free_fund")
        assert not hasattr(reloaded, "export_to_csv")
        assert not hasattr(reloaded, "export_data")

    importlib.reload(module)
