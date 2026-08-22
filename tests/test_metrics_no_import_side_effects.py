from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _drop_metrics_modules() -> None:
    for name in list(sys.modules):
        if name == "trend_analysis.metrics" or name.startswith("trend_analysis.metrics."):
            sys.modules.pop(name, None)


def test_metrics_import_does_not_patch_builtins(monkeypatch) -> None:
    monkeypatch.delattr(builtins, "annualize_return", raising=False)
    monkeypatch.delattr(builtins, "annualize_volatility", raising=False)
    _drop_metrics_modules()

    metrics = importlib.import_module("trend_analysis.metrics")

    assert not hasattr(metrics, "annualize_" + "return")
    assert not hasattr(metrics, "annualize_" + "volatility")
    assert not hasattr(builtins, "annualize_return")
    assert not hasattr(builtins, "annualize_volatility")


def test_metrics_import_does_not_restore_legacy_metric_oracle() -> None:
    legacy_module_name = "tests." + "legacy_metrics"
    sys.modules.pop(legacy_module_name, None)
    _drop_metrics_modules()

    importlib.import_module("trend_analysis.metrics")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_module_name)
