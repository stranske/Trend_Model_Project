import importlib
import importlib.metadata
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
