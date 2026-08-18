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


def test_lazy_proxy_import_uses_registered_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_proxy = types.ModuleType("trend_analysis.proxy")
    monkeypatch.setitem(sys.modules, "trend_analysis.proxy", stub_proxy)

    module = importlib.reload(trend_analysis)
    assert "proxy" not in module.__dict__
    assert module.proxy is stub_proxy
    assert module.__dict__["proxy"] is stub_proxy


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


def test_spec_proxy_name_restores_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(trend_analysis)
    proxy = module.__spec__
    assert isinstance(proxy, trend_analysis._SpecProxy)
    monkeypatch.delitem(sys.modules, "trend_analysis", raising=False)
    assert proxy.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is module


def test_spec_proxy_name_overwrites_foreign_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.reload(trend_analysis)
    proxy = module.__spec__
    sentinel = types.ModuleType("trend_analysis")
    monkeypatch.setitem(sys.modules, "trend_analysis", sentinel)

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
