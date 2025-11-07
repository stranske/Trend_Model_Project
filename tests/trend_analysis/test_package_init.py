"""Tests ensuring the top-level trend_analysis package wiring is covered."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, call, patch

import pytest

PACKAGE = "trend_analysis"


def _clear_trend_analysis_modules() -> None:
    """Remove ``trend_analysis`` modules from ``sys.modules`` for a clean import."""

    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(f"{PACKAGE}."):
            sys.modules.pop(name, None)


def _stub_module(name: str, **attrs: Any) -> SimpleNamespace:
    """Return a simple module-like object populated with ``attrs``."""

    module = SimpleNamespace(__name__=name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def stubbed_imports():
    """Provide a helper for reloading the package with patched imports."""

    real_import_module = importlib.import_module

    def _reload_with(
        stubs: dict[str, Any],
        *,
        version: str | None = "1.2.3",
        missing: set[str] | None = None,
    ):
        _clear_trend_analysis_modules()
        missing = set(missing or ())

        def fake_import(target: str, package: str | None = None):
            if target in missing:
                raise ImportError(target)
            if target in stubs:
                module = stubs[target]
                sys.modules.setdefault(target, module)
                return module
            return real_import_module(target, package=package)

        version_ctx: Callable[..., Any]
        if version is None:
            version_ctx = patch(
                "importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError,
            )
        else:
            version_ctx = patch("importlib.metadata.version", return_value=version)

        with patch("importlib.import_module", side_effect=fake_import) as import_mock:
            with version_ctx:
                module = importlib.import_module(PACKAGE)
        return module, import_mock

    return _reload_with


@pytest.fixture()
def package_stubs():
    """Return the base set of stub modules expected during package import."""

    eager = {
        f"{PACKAGE}.metrics": _stub_module(f"{PACKAGE}.metrics"),
        f"{PACKAGE}.config": _stub_module(f"{PACKAGE}.config"),
        f"{PACKAGE}.data": _stub_module(
            f"{PACKAGE}.data",
            identify_risk_free_fund=MagicMock(name="identify_risk_free_fund"),
            load_csv=MagicMock(name="load_csv"),
        ),
        f"{PACKAGE}.pipeline": _stub_module(f"{PACKAGE}.pipeline"),
        f"{PACKAGE}.export": _stub_module(
            f"{PACKAGE}.export",
            combined_summary_frame=MagicMock(name="combined_summary_frame"),
            combined_summary_result=MagicMock(name="combined_summary_result"),
            export_bundle=MagicMock(name="export_bundle"),
            export_data=MagicMock(name="export_data"),
            export_multi_period_metrics=MagicMock(name="export_multi_period_metrics"),
            export_phase1_multi_metrics=MagicMock(name="export_phase1_multi_metrics"),
            export_phase1_workbook=MagicMock(name="export_phase1_workbook"),
            export_to_csv=MagicMock(name="export_to_csv"),
            export_to_excel=MagicMock(name="export_to_excel"),
            export_to_json=MagicMock(name="export_to_json"),
            export_to_txt=MagicMock(name="export_to_txt"),
            flat_frames_from_results=MagicMock(name="flat_frames_from_results"),
            make_summary_formatter=MagicMock(name="make_summary_formatter"),
            metrics_from_result=MagicMock(name="metrics_from_result"),
            phase1_workbook_data=MagicMock(name="phase1_workbook_data"),
            register_formatter_excel=MagicMock(name="register_formatter_excel"),
            reset_formatters_excel=MagicMock(name="reset_formatters_excel"),
        ),
        f"{PACKAGE}.signals": _stub_module(f"{PACKAGE}.signals"),
        f"{PACKAGE}.backtesting": _stub_module(f"{PACKAGE}.backtesting"),
    }

    lazy = {
        f"{PACKAGE}.api": _stub_module(f"{PACKAGE}.api"),
        f"{PACKAGE}.cli": _stub_module(f"{PACKAGE}.cli"),
        f"{PACKAGE}.io": _stub_module(f"{PACKAGE}.io"),
        f"{PACKAGE}.selector": _stub_module(f"{PACKAGE}.selector"),
        f"{PACKAGE}.weighting": _stub_module(f"{PACKAGE}.weighting"),
        f"{PACKAGE}.weights": _stub_module(f"{PACKAGE}.weights"),
        f"{PACKAGE}.presets": _stub_module(f"{PACKAGE}.presets"),
        f"{PACKAGE}.run_multi_analysis": _stub_module(f"{PACKAGE}.run_multi_analysis"),
    }
    eager.update(lazy)
    return eager


def test_eager_submodules_imported(package_stubs, stubbed_imports):
    module, import_mock = stubbed_imports(package_stubs)

    for name in [
        "metrics",
        "config",
        "data",
        "pipeline",
        "export",
        "signals",
        "backtesting",
    ]:
        attr = getattr(module, name)
        assert attr is package_stubs[f"{PACKAGE}.{name}"]

    expected_calls = [
        call(f"{PACKAGE}.{name}")
        for name in [
            "metrics",
            "config",
            "data",
            "pipeline",
            "export",
            "signals",
            "backtesting",
        ]
    ]
    import_mock.assert_has_calls(expected_calls, any_order=True)

    assert (
        module.identify_risk_free_fund
        is package_stubs[f"{PACKAGE}.data"].identify_risk_free_fund
    )
    assert module.load_csv is package_stubs[f"{PACKAGE}.data"].load_csv
    assert module.export_to_csv is package_stubs[f"{PACKAGE}.export"].export_to_csv
    assert (
        module.combined_summary_frame
        is package_stubs[f"{PACKAGE}.export"].combined_summary_frame
    )


def test_lazy_imports_are_cached(package_stubs, stubbed_imports):
    module, _ = stubbed_imports(package_stubs)

    def fake_import(name: str, package: str | None = None):
        resolved = package_stubs[name]
        sys.modules.setdefault(name, resolved)
        return resolved

    lazy_names = [
        "api",
        "cli",
        "io",
        "selector",
        "weighting",
        "weights",
        "presets",
        "run_multi_analysis",
    ]

    with patch("importlib.import_module", side_effect=fake_import) as import_mock:
        first = getattr(module, "api")
        assert first is package_stubs[f"{PACKAGE}.api"]
        # Cached attribute should be returned on subsequent access without a new import
        assert module.api is first

        for name in lazy_names[1:]:
            getattr(module, name)

    expected = [call(f"{PACKAGE}.{name}") for name in lazy_names]
    import_mock.assert_has_calls(expected)


def test_missing_lazy_attribute_raises(package_stubs, stubbed_imports):
    module, _ = stubbed_imports(package_stubs)

    with pytest.raises(AttributeError):
        getattr(module, "not_a_module")


def test_metadata_version_fallback(package_stubs, stubbed_imports):
    module, _ = stubbed_imports(package_stubs, version=None)

    assert module.__version__ == "0.1.0-dev"


def test_metadata_version_from_distribution(package_stubs, stubbed_imports):
    module, _ = stubbed_imports(package_stubs, version="9.9.9")

    assert module.__version__ == "9.9.9"
    assert "__version__" in module.__all__


def test_optional_modules_absent(package_stubs, stubbed_imports):
    missing = {f"{PACKAGE}.data", f"{PACKAGE}.export"}
    module, _ = stubbed_imports(package_stubs, missing=missing)

    assert "data" not in module.__dict__
    assert "export" not in module.__dict__
    # Data/export helpers should not be exposed when modules fail to import
    assert not hasattr(module, "load_csv")
    assert not hasattr(module, "export_to_csv")


def test_dataclass_patch_recovers_missing_module(monkeypatch):
    """The dataclass guard should repopulate missing module entries."""

    import dataclasses
    import importlib
    import sys
    import typing
    from types import ModuleType
    from typing import ClassVar

    module = importlib.reload(dataclasses)
    module.__dict__.pop("_trend_model_patched", None)

    from trend_analysis import _patch_dataclasses_module_guard

    _patch_dataclasses_module_guard()

    class Dummy:
        """Sentinel class whose module entry will be removed."""

        pass

    Dummy.__module__ = "ghost.module"
    monkeypatch.delitem(sys.modules, "ghost.module", raising=False)

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "ghost.module":
            raise ImportError(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = dataclasses._is_type(  # type: ignore[attr-defined]
        "ClassVar",
        Dummy,
        typing,
        ClassVar,
        lambda candidate, mod: candidate is ClassVar,
    )
    assert result is False

    restored = sys.modules["ghost.module"]
    assert isinstance(restored, ModuleType)
    assert restored.__dict__["__package__"] == "ghost"


def test_spec_proxy_reinstates_module_registration(monkeypatch):
    """Accessing ``name`` should restore the module in ``sys.modules``."""

    import sys
    from types import SimpleNamespace

    import trend_analysis as package

    proxy = package._SpecProxy(SimpleNamespace(name="trend_analysis"))

    monkeypatch.delitem(sys.modules, "trend_analysis", raising=False)
    assert proxy.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is package
