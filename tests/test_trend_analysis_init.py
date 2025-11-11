"""Tests for the ``trend_analysis`` package initializer."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType
from typing import Iterator

import pytest


def _purge_trend_analysis_modules() -> None:
    """Remove cached ``trend_analysis`` modules from ``sys.modules``.

    Reloading the package is required for many of the behaviours under test, so
    this helper clears both the package itself and any already-imported
    submodules to ensure each test exercises the initializer logic from a clean
    state.
    """

    for name in list(sys.modules):
        if name == "trend_analysis" or name.startswith("trend_analysis."):
            sys.modules.pop(name)


@pytest.fixture(autouse=True)
def _fresh_trend_analysis() -> Iterator[None]:
    """Guarantee a clean import state before and after each test."""

    _purge_trend_analysis_modules()
    try:
        yield
    finally:
        _purge_trend_analysis_modules()


def test_dataclass_guard_recovers_missing_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The patched ``dataclasses._is_type`` restores removed module entries."""

    import dataclasses

    module = importlib.import_module("trend_analysis")

    original_safe = dataclasses._is_type  # type: ignore[attr-defined]

    def stub_is_type(
        annotation: object,
        cls: type[object],
        a_module: object,
        a_type: object,
        predicate: object,
    ) -> bool:
        if cls.__module__ not in sys.modules:
            raise AttributeError("module missing")
        return bool(original_safe(annotation, cls, a_module, a_type, predicate))

    monkeypatch.setattr(dataclasses, "_is_type", stub_is_type, raising=False)
    monkeypatch.setattr(dataclasses, "_trend_model_patched", False, raising=False)

    module._patch_dataclasses_module_guard()  # type: ignore[attr-defined]
    patched = dataclasses._is_type  # type: ignore[attr-defined]
    missing_name = "tests.fake_trend_analysis_guard"
    sys.modules.pop(missing_name, None)
    dummy_cls = type("Dummy", (), {"__module__": missing_name})

    try:
        result = patched("Dummy", dummy_cls, None, None, lambda *_: False)
        assert isinstance(result, bool)
        placeholder = sys.modules[missing_name]
        assert isinstance(placeholder, ModuleType)
        assert placeholder.__package__ == "tests"
        assert module._SAFE_IS_TYPE is patched  # type: ignore[attr-defined]
        assert getattr(dataclasses, "_trend_model_patched") is True
        second = patched("Dummy", dummy_cls, None, None, lambda *_: False)
        assert isinstance(second, bool)
    finally:
        sys.modules.pop(missing_name, None)

    # A second invocation should reuse the cached safe handler without work.
    module._patch_dataclasses_module_guard()  # type: ignore[attr-defined]
    assert dataclasses._is_type is patched  # type: ignore[attr-defined]


def test_spec_proxy_re_registers_module() -> None:
    """Accessing the spec proxies ``sys.modules`` to the live module object."""

    module = importlib.import_module("trend_analysis")
    spec = module.__spec__
    assert spec is not None
    assert spec.name == "trend_analysis"

    sys.modules.pop("trend_analysis")
    assert "trend_analysis" not in sys.modules
    assert module.__spec__.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is module

    sys.modules.pop("trend_analysis")
    module._ensure_registered()  # type: ignore[attr-defined]
    assert sys.modules["trend_analysis"] is module


def test_lazy_loader_imports_and_caches_modules() -> None:
    """Lazy attributes import their modules once and remain cached."""

    module = importlib.import_module("trend_analysis")

    module.__dict__.pop("api", None)
    sys.modules.pop("trend_analysis.api", None)

    api_module = module.api
    assert api_module is sys.modules["trend_analysis.api"]
    assert module.api is api_module

    with pytest.raises(AttributeError):
        getattr(module, "definitely_missing")


def test_version_metadata_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package metadata populates ``__version__`` when available."""

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    module = importlib.import_module("trend_analysis")
    assert module.__version__ == "9.9.9"


def test_version_metadata_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing distribution falls back to the development version string."""

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    module = importlib.import_module("trend_analysis")
    assert module.__version__ == "0.1.0-dev"


def test_dataclass_guard_handles_absent_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the guard without ``_is_type`` leaves the patch untouched."""

    import dataclasses

    module = importlib.import_module("trend_analysis")
    monkeypatch.delattr(dataclasses, "_is_type", raising=False)
    module._patch_dataclasses_module_guard()  # type: ignore[attr-defined]
    assert not hasattr(dataclasses, "_is_type")


def test_initializer_branches_skip_when_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing with missing optional modules executes the skip branches."""

    from importlib import util
    from pathlib import Path

    missing = {"trend_analysis.data", "trend_analysis.export"}
    real_import = importlib.import_module

    def guarded(name: str, package: str | None = None) -> ModuleType:
        if name in missing:
            raise ImportError("optional dependency unavailable")
        if name.startswith("trend_analysis.") and name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return real_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", guarded)

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "trend_analysis" / "__init__.py"
    )
    spec = util.spec_from_file_location("trend_analysis", module_path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    module.__spec__ = None
    sys.modules["trend_analysis"] = module
    spec.loader.exec_module(module)

    assert "data" not in module.__dict__
    assert "export" not in module.__dict__
    assert module._ORIGINAL_SPEC is None  # type: ignore[attr-defined]


def test_dataclass_guard_rejects_missing_module_name() -> None:
    """The safe guard propagates AttributeError when no module is available."""

    module = importlib.import_module("trend_analysis")
    missing_module_cls = type("NoModule", (), {"__module__": None})

    with pytest.raises(AttributeError):
        module._SAFE_IS_TYPE("Dummy", missing_module_cls, None, None, lambda *_: False)  # type: ignore[attr-defined]
