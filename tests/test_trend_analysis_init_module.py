from __future__ import annotations

import importlib
import inspect
import sys
from importlib import metadata
from types import ModuleType
from typing import Callable

import pytest

import trend_analysis


# Compatibility helper: Python 3.12 added `module=` to dataclasses.make_dataclass.
# Make tests work on older runtimes (3.11 CI) by emulating that behaviour when
# the keyword is not available.
def _make_dataclass_with_module(
    name: str, fields: list[tuple[str, type]], module: str | None
) -> type:
    import dataclasses

    sig = inspect.signature(dataclasses.make_dataclass)
    kwargs = {}
    if "module" in sig.parameters and module is not None:
        kwargs["module"] = module

    # Create the dataclass (passing module via kwargs on runtimes that
    # support it). For older stdlib versions the kwargs will be empty and
    # we'll attach the __module__ and a placeholder in sys.modules below.
    cls = dataclasses.make_dataclass(name, fields, **kwargs)
    if module is not None:
        cls.__module__ = module
        sys.modules.setdefault(module, ModuleType(module))
    return cls


def _reload_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data_funcs: dict[str, Callable],
    export_funcs: dict[str, Callable],
) -> ModuleType:
    """Reload ``trend_analysis`` after priming stub submodules.

    The package's ``__init__`` eagerly imports a curated list of submodules and
    then conditionally re-exports helpers from ``data`` and ``export``.  The
    helper clears any previously imported package state, injects lightweight
    stand-ins for the required submodules, and finally imports the top-level
    package so the conditional wiring runs against the controlled environment.
    """

    for name in list(sys.modules):
        if name == "trend_analysis" or name.startswith("trend_analysis."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    def register_stub(name: str, attrs: dict[str, Callable] | None = None) -> None:
        module = ModuleType(f"trend_analysis.{name}")
        for attr, value in (attrs or {}).items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    for eager_name in [
        "metrics",
        "config",
        "pipeline",
        "signals",
        "backtesting",
    ]:
        register_stub(eager_name)

    register_stub("data", data_funcs)
    register_stub("export", export_funcs)

    for lazy_name in [
        "io",
        "selector",
        "weighting",
        "weights",
        "presets",
        "engine",
        "perf",
        "regimes",
        "multi_period",
        "plugins",
        "proxy",
    ]:
        register_stub(lazy_name)

    # Arrange for importlib.metadata.version to raise PackageNotFoundError so
    # the package fallback path in the top-level ``trend_analysis`` module is
    # exercised during tests.
    monkeypatch.setattr(
        metadata,
        "version",
        lambda _: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
        raising=False,
    )

    return importlib.import_module("trend_analysis")


@pytest.fixture
def reload_trend_analysis():
    module = trend_analysis

    def _reload() -> ModuleType:
        return importlib.reload(module)

    yield _reload
    importlib.reload(module)


def test_spec_proxy_reregisters_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = ModuleType("trend_analysis")
    monkeypatch.setitem(sys.modules, "trend_analysis", sentinel)

    name = trend_analysis.__spec__.name  # type: ignore[union-attr]
    assert name == "trend_analysis"
    assert sys.modules["trend_analysis"] is trend_analysis


def test_lazy_attribute_loader_imports_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(trend_analysis.__dict__, "proxy", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.proxy", raising=False)

    proxy_module = getattr(trend_analysis, "proxy")
    assert proxy_module is sys.modules["trend_analysis.proxy"]
    assert trend_analysis.proxy is proxy_module


def test_lazy_attribute_loader_unknown_attr() -> None:
    with pytest.raises(AttributeError):
        trend_analysis.__getattr__("not_a_module")


def test_version_fallback_when_package_metadata_missing(
    monkeypatch: pytest.MonkeyPatch, reload_trend_analysis
) -> None:
    from importlib import metadata

    def _raise_package_not_found(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise_package_not_found, raising=False)

    module = reload_trend_analysis()
    assert module.__version__ == "0.1.0-dev"


def test_eager_import_skips_missing_submodule(
    monkeypatch: pytest.MonkeyPatch, reload_trend_analysis
) -> None:
    original_import = importlib.import_module

    def _patched(name: str, package: str | None = None):
        if name == "trend_analysis.metrics":
            raise ImportError("metrics unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched, raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.metrics", raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "metrics", raising=False)

    module = reload_trend_analysis()
    assert "metrics" not in module.__dict__


def test_conditional_exports_omitted_when_dependencies_fail(
    monkeypatch: pytest.MonkeyPatch, reload_trend_analysis
) -> None:
    original_import = importlib.import_module

    def _patched(name: str, package: str | None = None):
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("dependency missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched, raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.data", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.export", raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "data", raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "export", raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "identify_risk_free_fund", raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "export_data", raising=False)

    module = reload_trend_analysis()
    assert "identify_risk_free_fund" not in module.__dict__
    assert "export_data" not in module.__dict__
