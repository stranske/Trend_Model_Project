import importlib
import sys
from types import ModuleType

import pytest


def test_lazy_attribute_loading(monkeypatch):
    module = importlib.import_module("trend_analysis")
    fake_module = ModuleType("trend_analysis.fake_module")
    fake_module.value = 42
    monkeypatch.setitem(sys.modules, "trend_analysis.fake_module", fake_module)
    monkeypatch.setitem(module._LAZY_SUBMODULES, "fake", "trend_analysis.fake_module")
    module.__dict__.pop("fake", None)

    result = module.fake
    assert result is fake_module
    assert module.__dict__["fake"] is fake_module


def test_unknown_lazy_attribute_raises():
    module = importlib.import_module("trend_analysis")
    with pytest.raises(AttributeError):
        _ = module.__getattr__("does_not_exist")


def test_version_fallback(monkeypatch):
    def raise_not_found(name):  # pragma: no cover - defensive helper
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    original = sys.modules.get("trend_analysis")
    sys.modules.pop("trend_analysis", None)
    try:
        module = importlib.import_module("trend_analysis")
        assert module.__version__ == "0.1.0-dev"
    finally:
        if original is not None:
            sys.modules["trend_analysis"] = original
            importlib.reload(original)


@pytest.mark.usefixtures("restore_trend_analysis")
def test_data_exports_available():
    module = importlib.import_module("trend_analysis")
    assert module.load_csv is module.data.load_csv
    assert module.identify_risk_free_fund is module.data.identify_risk_free_fund


def test_export_reexports_available():
    module = importlib.import_module("trend_analysis")
    assert module.export_to_json is module.export.export_to_json
    assert module.register_formatter_excel is module.export.register_formatter_excel


def test_eager_import_failure_is_skipped(monkeypatch):
    import trend_analysis as original_module

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "trend_analysis.metrics":
            raise ImportError("optional metrics missing")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop("trend_analysis", None)
    try:
        module = importlib.import_module("trend_analysis")
        assert not hasattr(module, "metrics")
    finally:
        sys.modules["trend_analysis"] = original_module
        importlib.reload(original_module)


@pytest.fixture
def restore_trend_analysis(monkeypatch):
    module = importlib.import_module("trend_analysis")
    yield
    importlib.reload(module)
