import dataclasses
import importlib
import sys
import types

import pytest

import trend_analysis


@pytest.fixture(autouse=True)
def _reset_trend_analysis():
    """Ensure ``trend_analysis`` reloads after each test."""
    yield
    importlib.reload(trend_analysis)


def test_lazy_cli_import_uses_registered_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cli = types.ModuleType("trend_analysis.cli")
    monkeypatch.setitem(sys.modules, "trend_analysis.cli", stub_cli)

    module = importlib.reload(trend_analysis)
    assert "cli" not in module.__dict__
    assert module.cli is stub_cli
    assert module.__dict__["cli"] is stub_cli


def test_unknown_attribute_raises_attribute_error() -> None:
    module = importlib.reload(trend_analysis)
    with pytest.raises(AttributeError):
        module.__getattr__("not_a_real_module")


def test_version_fallback_used_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_missing(name: str) -> str:  # noqa: ANN001
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", _raise_missing)
    module = importlib.reload(trend_analysis)
    assert module.__version__ == "0.1.0-dev"


def test_dataclasses_patch_recreates_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_is_type(annotation, cls, a_module, a_type, predicate):  # noqa: ANN001
        if cls.__module__ not in sys.modules:
            raise AttributeError("missing module")
        return True

    monkeypatch.setattr(dataclasses, "_is_type", fake_is_type, raising=False)
    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)

    module = importlib.reload(trend_analysis)

    class Ghost:
        __module__ = "missing.module"

    monkeypatch.delitem(sys.modules, Ghost.__module__, raising=False)

    def failing_import(name: str, package: str | None = None):  # noqa: ANN001
        if name == Ghost.__module__:
            raise ImportError("no module")
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", failing_import)

    result = dataclasses._is_type(None, Ghost, None, None, lambda _: False)
    assert result is True
    created = sys.modules[Ghost.__module__]
    assert isinstance(created, types.ModuleType)
    assert created.__package__ == "missing"


def test_dataclasses_patch_bubbles_when_module_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_missing(annotation, cls, a_module, a_type, predicate):  # noqa: ANN001
        raise AttributeError("missing module")

    monkeypatch.setattr(dataclasses, "_is_type", always_missing, raising=False)
    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)

    module = importlib.reload(trend_analysis)

    class Nameless:
        __module__ = None

    with pytest.raises(AttributeError):
        dataclasses._is_type(None, Nameless, None, None, lambda _: False)


def test_spec_proxy_name_restores_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(trend_analysis)
    proxy = module.__spec__
    assert isinstance(proxy, trend_analysis._SpecProxy)
    monkeypatch.delitem(sys.modules, "trend_analysis", raising=False)
    assert proxy.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is module


def test_eager_import_skips_missing_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):  # noqa: ANN001
        if name == "trend_analysis.export":
            raise ImportError("optional dependency missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.delitem(trend_analysis.__dict__, "export", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.export", raising=False)
    module = importlib.reload(trend_analysis)
    assert "export" not in module.__dict__
    with pytest.raises(AttributeError):
        module.__getattr__("export")


def test_dataclasses_patch_returns_when_module_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_is_type(annotation, cls, a_module, a_type, predicate):  # noqa: ANN001
        return True

    monkeypatch.setattr(dataclasses, "_is_type", fake_is_type, raising=False)
    monkeypatch.delattr(dataclasses, "_trend_model_patched", raising=False)

    module = importlib.reload(trend_analysis)

    class Existing:
        __module__ = "trend_analysis"

    assert dataclasses._is_type(None, Existing, None, None, lambda _: False)
