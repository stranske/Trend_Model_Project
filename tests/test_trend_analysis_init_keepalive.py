"""Focused coverage for ``trend_analysis.__init__`` helper logic."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Callable

import pytest


def reload_trend_analysis() -> ModuleType:
    """Reload ``trend_analysis`` without disturbing unrelated modules."""

    module = importlib.import_module("trend_analysis")
    return importlib.reload(module)


@pytest.fixture(autouse=True)
def reset_dataclasses() -> None:
    module = importlib.reload(importlib.import_module("dataclasses"))
    module.__dict__.pop("_trend_model_patched", None)
    yield
    importlib.reload(importlib.import_module("dataclasses"))


@pytest.fixture
def fresh_dataclasses() -> Callable[[], ModuleType]:
    """Return a factory that yields a freshly reloaded ``dataclasses`` module."""

    def _factory() -> ModuleType:
        module = importlib.reload(importlib.import_module("dataclasses"))
        module.__dict__.pop("_trend_model_patched", None)
        return module

    return _factory


def test_patch_guard_reimports_missing_module(fresh_dataclasses: Callable[[], ModuleType]) -> None:
    module = reload_trend_analysis()
    dataclasses_module = fresh_dataclasses()
    original = dataclasses_module._is_type  # type: ignore[attr-defined]
    calls: list[str] = []

    def stub(annotation: Any, cls: Any, *_args: Any, **_kwargs: Any) -> bool:
        module_name = getattr(cls, "__module__", None)
        if module_name == "nonexistent.module" and module_name not in sys.modules:
            calls.append(module_name)
            raise AttributeError("missing module")
        if module_name == "nonexistent.module":
            return True
        return bool(original(annotation, cls, *_args, **_kwargs))

    dataclasses_module._is_type = stub  # type: ignore[attr-defined]
    module._patch_dataclasses_module_guard()
    patched = importlib.import_module("dataclasses")._is_type  # type: ignore[attr-defined]

    fake_type = type("MissingType", (), {})
    fake_type.__module__ = "nonexistent.module"  # type: ignore[attr-defined]
    sys.modules.pop("nonexistent.module", None)

    assert patched(None, fake_type, None, None, None) is True
    assert calls == ["nonexistent.module"]
    placeholder = sys.modules["nonexistent.module"]
    assert isinstance(placeholder, ModuleType)
    assert placeholder.__package__ == "nonexistent"


def test_patch_guard_is_idempotent(fresh_dataclasses: Callable[[], ModuleType]) -> None:
    fresh_dataclasses()
    module = reload_trend_analysis()
    patched_before = importlib.import_module("dataclasses")._is_type  # type: ignore[attr-defined]

    module._patch_dataclasses_module_guard()
    assert importlib.import_module("dataclasses")._is_type is patched_before


def test_patch_guard_reraises_for_missing_module_name(fresh_dataclasses: Callable[[], ModuleType]) -> None:
    module = reload_trend_analysis()
    dataclasses_module = fresh_dataclasses()

    def stub(annotation: Any, cls: Any, *_args: Any, **_kwargs: Any) -> bool:
        raise AttributeError("boom")

    dataclasses_module._is_type = stub  # type: ignore[attr-defined]
    module._patch_dataclasses_module_guard()
    patched = importlib.import_module("dataclasses")._is_type  # type: ignore[attr-defined]

    class NoModule:
        __module__ = ""

    with pytest.raises(AttributeError):
        patched("typing.ClassVar[int]", NoModule, None, None, None)


def test_spec_proxy_restores_module_registration() -> None:
    module = reload_trend_analysis()
    spec = getattr(module, "__spec__")
    assert spec.name == "trend_analysis"

    sys.modules[module.__name__] = ModuleType("trend_analysis_shadow")
    assert spec.name == "trend_analysis"
    assert sys.modules[module.__name__] is module


def test_eager_import_skips_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "trend_analysis.pipeline":
            raise ImportError("optional")
        return original_import_module(name, package)

    module = reload_trend_analysis()
    module.__dict__.pop("pipeline", None)
    sys.modules.pop("trend_analysis.pipeline", None)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    module = reload_trend_analysis()
    assert "pipeline" not in module.__dict__


def test_lazy_getattr_imports_and_caches() -> None:
    module = reload_trend_analysis()
    sys.modules.pop("trend_analysis.plugins", None)
    module.__dict__.pop("plugins", None)

    plugins = module.__getattr__("plugins")
    assert plugins is sys.modules["trend_analysis.plugins"]
    assert module.plugins is plugins

    with pytest.raises(AttributeError):
        module.__getattr__("does_not_exist")


def test_reexports_available_when_submodules_present() -> None:
    module = reload_trend_analysis()
    assert hasattr(module, "load_csv")
    assert hasattr(module, "identify_risk_free_fund")
    assert hasattr(module, "export_data")
    assert hasattr(module, "register_formatter_excel")


def test_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", raise_missing)
    module = reload_trend_analysis()
    assert module.__version__ == "0.1.0-dev"


