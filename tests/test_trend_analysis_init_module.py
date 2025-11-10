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
        "run_multi_analysis",
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
def restore_dataclasses(monkeypatch: pytest.MonkeyPatch):
    import dataclasses
    import typing

    original_is_type = getattr(dataclasses, "_is_type", None)
    original_flag = getattr(dataclasses, "_trend_model_patched", False)
    if original_flag:
        monkeypatch.delattr(dataclasses, "_trend_model_patched")
    yield dataclasses
    monkeypatch.setattr(dataclasses, "_is_type", original_is_type, raising=False)
    if original_flag:
        monkeypatch.setattr(dataclasses, "_trend_model_patched", True, raising=False)


@pytest.fixture
def reload_trend_analysis():
    import dataclasses

    module = trend_analysis

    def _reload() -> ModuleType:
        if getattr(dataclasses, "_trend_model_patched", False):
            delattr(dataclasses, "_trend_model_patched")
        return importlib.reload(module)

    yield _reload

    if getattr(dataclasses, "_trend_model_patched", False):
        delattr(dataclasses, "_trend_model_patched")
    importlib.reload(module)

    def fake_import(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(name)

def test_patch_dataclasses_module_guard_reimports_missing_module(
    monkeypatch: pytest.MonkeyPatch, restore_dataclasses
) -> None:
    import dataclasses

    sentinel_name = "tests.synthetic_dataclass_module"
    dataclass_module = ModuleType(sentinel_name)
    sys.modules[sentinel_name] = dataclass_module

    dataclass_type = _make_dataclass_with_module(
        "Synth", [("value", int)], sentinel_name
    )
    sys.modules.pop(sentinel_name)

    call_order: list[ModuleType | None] = []

    def _probe_is_type(*args: object) -> bool:
        call_order.append(sys.modules.get(sentinel_name))
        module_present = call_order[-1] is not None
        if not module_present:
            raise AttributeError("module missing")
        return True

    monkeypatch.setattr(dataclasses, "_is_type", _probe_is_type, raising=False)

    def fake_import(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(name)

    assert getattr(dataclasses, "_trend_model_patched", False) is True

    result = dataclasses._is_type(  # type: ignore[attr-defined]
        annotation=None,
        cls=dataclass_type,
        a_module=None,
        a_type=None,
        predicate=None,
    )
    assert result is True
    assert call_order == [None, sys.modules[sentinel_name]]
    assert isinstance(sys.modules[sentinel_name], ModuleType)
    assert getattr(sys.modules[sentinel_name], "__package__", None) == "tests"


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


def test_patch_guard_skips_when_original_missing(
    monkeypatch: pytest.MonkeyPatch, restore_dataclasses
) -> None:
    import dataclasses

    monkeypatch.delattr(dataclasses, "_is_type", raising=False)
    trend_analysis._patch_dataclasses_module_guard()
    assert not hasattr(dataclasses, "_trend_model_patched")


def test_patch_guard_propagates_when_module_name_missing(
    monkeypatch: pytest.MonkeyPatch, restore_dataclasses
) -> None:
    import dataclasses

    dataclass_type = _make_dataclass_with_module(
        "Nameless", [("value", int)], "tests.nameless"
    )
    dataclass_type.__module__ = ""

    def _probe(*_: object) -> bool:
        raise AttributeError("module missing")

    monkeypatch.setattr(dataclasses, "_is_type", _probe, raising=False)
    trend_analysis._patch_dataclasses_module_guard()

    with pytest.raises(AttributeError):
        # Arguments: instance, cls, args, kwargs, module
        dataclasses._is_type(None, dataclass_type, None, None, None)  # type: ignore[attr-defined]

    assert name == "trend_analysis"
    assert sys.modules["trend_analysis"] is trend_analysis_module

def test_patch_guard_retries_when_module_already_loaded(
    monkeypatch: pytest.MonkeyPatch, restore_dataclasses
) -> None:
    import dataclasses

    sentinel_name = "tests.preloaded_dataclass_module"
    module = ModuleType(sentinel_name)
    sys.modules[sentinel_name] = module

    dataclass_type = _make_dataclass_with_module(
        "Preloaded", [("value", int)], sentinel_name
    )

    call_counter = {"count": 0}

    def _probe(*_: object) -> bool:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise AttributeError("module missing")
        return True

    monkeypatch.setattr(dataclasses, "_is_type", _probe, raising=False)
    trend_analysis._patch_dataclasses_module_guard()

    # _is_type signature: (obj, cls, a, b, c)
    # Here: obj=None, cls=dataclass_type, a=None, b=None, c=None
    assert dataclasses._is_type(None, dataclass_type, None, None, None) is True  # type: ignore[attr-defined]
    assert call_counter["count"] == 2
    assert sys.modules[sentinel_name] is module


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
    monkeypatch.delitem(
        trend_analysis.__dict__, "identify_risk_free_fund", raising=False
    )
    monkeypatch.delitem(trend_analysis.__dict__, "export_data", raising=False)

    module = reload_trend_analysis()
    assert "identify_risk_free_fund" not in module.__dict__
    assert "export_data" not in module.__dict__
