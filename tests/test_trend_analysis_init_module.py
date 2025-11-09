"""Tests for the high-level package bootstrap in ``trend_analysis.__init__``."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest


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

    monkeypatch.setattr(
        metadata,
        "version",
        lambda _: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
        raising=False,
    )

    return importlib.import_module("trend_analysis")


def test_dataclasses_guard_reimports_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    import trend_analysis

    calls: list[str] = []

    def record(name: str) -> ModuleType:
        calls.append(name)
        return ModuleType(name)

    monkeypatch.setattr(trend_analysis.importlib, "import_module", record)

    state: list[object | None] = []

    def stub(annotation: str, cls: type, *_: object) -> bool:
        state.append(sys.modules.get(cls.__module__))
        if len(state) == 1:
            raise AttributeError("missing module")
        return True

    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)
    monkeypatch.setattr(dataclasses, "_is_type", stub, raising=False)

    trend_analysis._patch_dataclasses_module_guard()

    dummy = type("Dummy", (), {})
    dummy.__module__ = "pkg.missing"
    sys.modules.pop("pkg.missing", None)

    assert dataclasses._is_type("pkg.missing.Dummy", dummy, None, None, lambda _: False)
    assert calls == ["pkg.missing"]
    assert isinstance(sys.modules["pkg.missing"], ModuleType)
    assert state[0] is None
    assert state[1] is sys.modules["pkg.missing"]


def test_dataclasses_guard_fallback_creates_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    import trend_analysis

    def boom(_: str) -> ModuleType:  # pragma: no cover - signature matches importlib
        raise RuntimeError("boom")

    monkeypatch.setattr(trend_analysis.importlib, "import_module", boom)

    history: list[object | None] = []

    def stub(annotation: str, cls: type, *_: object) -> bool:
        history.append(sys.modules.get(cls.__module__))
        if len(history) == 1:
            raise AttributeError("missing module")
        return False

    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)
    monkeypatch.setattr(dataclasses, "_is_type", stub, raising=False)

    trend_analysis._patch_dataclasses_module_guard()

    dummy = type("Fallback", (), {})
    dummy.__module__ = "pkg.placeholder"
    sys.modules.pop("pkg.placeholder", None)

    assert not dataclasses._is_type(
        "pkg.placeholder.Fallback", dummy, None, None, lambda _: True
    )
    module = sys.modules["pkg.placeholder"]
    assert isinstance(module, ModuleType)
    assert module.__package__ == "pkg"
    assert history[0] is None
    assert history[1] is module


def test_spec_proxy_re_registers_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    module_name = trend_analysis.__name__
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    proxy = trend_analysis._SpecProxy(SimpleNamespace(name=module_name))

    assert proxy.name == module_name
    assert sys.modules[module_name] is trend_analysis


def test_conditional_imports_bind_export_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_calls: list[tuple[str, dict[str, object]]] = []

    def fake_load_csv(path: str, **kwargs: object) -> dict[str, object]:
        data_calls.append((path, kwargs))
        return {"path": path, **kwargs}

    def make_export_func(name: str) -> Callable:
        return lambda *args, **kwargs: (name, args, kwargs)

    export_funcs = {
        key: make_export_func(key)
        for key in [
            "combined_summary_frame",
            "combined_summary_result",
            "export_bundle",
            "export_data",
            "export_multi_period_metrics",
            "export_phase1_multi_metrics",
            "export_phase1_workbook",
            "export_to_csv",
            "export_to_excel",
            "export_to_json",
            "export_to_txt",
            "flat_frames_from_results",
            "make_summary_formatter",
            "metrics_from_result",
            "phase1_workbook_data",
            "register_formatter_excel",
            "reset_formatters_excel",
        ]
    }

    module = _reload_with_stubs(
        monkeypatch,
        data_funcs={
            "identify_risk_free_fund": lambda *a, **kw: ("risk", a, kw),
            "load_csv": fake_load_csv,
        },
        export_funcs=export_funcs,
    )

    assert module.load_csv("sample.csv") == {"path": "sample.csv"}
    assert module.identify_risk_free_fund()[0] == "risk"
    assert module.export_to_json({"payload": 1})[0] == "export_to_json"
    assert module.__version__ == "0.1.0-dev"
    assert data_calls == [("sample.csv", {})]
    assert module.__spec__.name == "trend_analysis"
    assert set(export_funcs) <= set(module.__all__)


def test_dataclasses_guard_propagates_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    import trend_analysis

    def stub(annotation: str, cls: type, *_: object) -> bool:
        raise AttributeError("missing module reference")

    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)
    monkeypatch.setattr(dataclasses, "_is_type", stub, raising=False)

    trend_analysis._patch_dataclasses_module_guard()

    ghost = type("Ghost", (), {})
    ghost.__module__ = ""

    with pytest.raises(AttributeError):
        dataclasses._is_type("Ghost", ghost, None, None, lambda _: False)


def test_eager_import_skips_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    for name in list(sys.modules):
        if name == "trend_analysis" or name.startswith("trend_analysis."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    def register_stub(name: str, attrs: dict[str, object] | None = None) -> None:
        module = ModuleType(f"trend_analysis.{name}")
        for attr, value in (attrs or {}).items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    for eager_name in ["metrics", "config", "signals", "backtesting"]:
        register_stub(eager_name)

    register_stub(
        "data",
        {
            "identify_risk_free_fund": lambda *a, **kw: None,
            "load_csv": lambda *a, **kw: None,
        },
    )
    export_attrs = {
        name: (lambda *a, **kw: None)
        for name in [
            "combined_summary_frame",
            "combined_summary_result",
            "export_bundle",
            "export_data",
            "export_multi_period_metrics",
            "export_phase1_multi_metrics",
            "export_phase1_workbook",
            "export_to_csv",
            "export_to_excel",
            "export_to_json",
            "export_to_txt",
            "flat_frames_from_results",
            "make_summary_formatter",
            "metrics_from_result",
            "phase1_workbook_data",
            "register_formatter_excel",
            "reset_formatters_excel",
        ]
    }
    register_stub("export", export_attrs)

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        if name == "trend_analysis":
            return original_import(name, package)
        if name == "trend_analysis.pipeline":
            raise ImportError("pipeline missing")
        if name.startswith("trend_analysis.") and name in sys.modules:
            return sys.modules[name]
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(metadata, "version", lambda _: "1.2.3", raising=False)

    module = importlib.import_module("trend_analysis")

    assert not hasattr(module, "pipeline")
    assert module.metrics is sys.modules["trend_analysis.metrics"]
