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


def test_lazy_cli_import_uses_registered_module(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_version_fallback_used_when_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_missing(name: str) -> str:  # noqa: ANN001
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", _raise_missing)
    module = importlib.reload(trend_analysis)
    assert module.__version__ == "0.1.0-dev"
