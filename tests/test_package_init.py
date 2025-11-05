"""Regression tests for the package-level initialisation helpers."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def reload_trend_analysis() -> object:
    """Reload ``trend_analysis`` so tests observe fresh module state."""

    def _reload() -> object:
        sys.modules.pop("trend_analysis", None)
        module = importlib.import_module("trend_analysis")
        return module

    # Ensure the module is loaded for the first use in the test and yield
    module = _reload()
    try:
        yield module
    finally:
        # Leave the re-imported module available for follow-on tests
        sys.modules["trend_analysis"] = module


def test_eager_imports_expose_core_helpers(reload_trend_analysis: object) -> None:
    trend_analysis = reload_trend_analysis
    metrics_mod = importlib.import_module("trend_analysis.metrics")
    data_mod = importlib.import_module("trend_analysis.data")

    assert trend_analysis.metrics is metrics_mod
    assert trend_analysis.data is data_mod
    # ``load_csv`` and ``identify_risk_free_fund`` are re-exported when the
    # ``data`` module loads successfully.
    assert trend_analysis.load_csv is data_mod.load_csv
    assert trend_analysis.identify_risk_free_fund is data_mod.identify_risk_free_fund
    # The package defines a stable public surface via ``__all__``
    assert {"load_csv", "export_data", "metrics"}.issubset(set(trend_analysis.__all__))


def test_lazy_attribute_loads_requested_module(reload_trend_analysis: object) -> None:
    trend_analysis = reload_trend_analysis
    # Ensure the lazy module must be imported on access.
    sys.modules.pop("trend_analysis.cli", None)

    cli_mod = trend_analysis.__getattr__("cli")

    assert cli_mod is importlib.import_module("trend_analysis.cli")
    # Subsequent lookups should reuse the cached attribute on the package.
    assert trend_analysis.cli is cli_mod


def test_unknown_attribute_raises_attribute_error(
    reload_trend_analysis: object,
) -> None:
    trend_analysis = reload_trend_analysis

    with pytest.raises(AttributeError):
        trend_analysis.__getattr__("does_not_exist")


def test_missing_optional_submodules_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in {"trend_analysis.data", "trend_analysis.export"}:
            raise ImportError("simulated optional dependency chain failure")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop("trend_analysis", None)

    module = importlib.import_module("trend_analysis")

    assert "data" not in module.__dict__
    assert "export" not in module.__dict__
    assert not hasattr(module, "load_csv")

    importlib.reload(module)


def test_version_fallback_populates_dev_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module

    def passthrough(name: str, package: str | None = None):
        return original_import_module(name, package)

    def raise_package_not_found(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_package_not_found)
    monkeypatch.setattr(importlib, "import_module", passthrough)
    sys.modules.pop("trend_analysis", None)

    module = importlib.import_module("trend_analysis")

    assert module.__version__ == "0.1.0-dev"

    importlib.reload(module)
