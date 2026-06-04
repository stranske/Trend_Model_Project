from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import ModuleType


def _drop_metrics_modules() -> None:
    for name in list(sys.modules):
        if name == "trend_analysis.metrics" or name.startswith("trend_analysis.metrics."):
            sys.modules.pop(name, None)


def test_metrics_import_does_not_patch_builtins(monkeypatch) -> None:
    monkeypatch.delattr(builtins, "annualize_return", raising=False)
    monkeypatch.delattr(builtins, "annualize_volatility", raising=False)
    _drop_metrics_modules()

    metrics = importlib.import_module("trend_analysis.metrics")

    assert metrics.annualize_return is metrics.annual_return
    assert metrics.annualize_volatility is metrics.volatility
    assert not hasattr(builtins, "annualize_return")
    assert not hasattr(builtins, "annualize_volatility")


def test_metrics_import_does_not_replace_real_legacy_metrics_module() -> None:
    sys.modules.pop("tests.legacy_metrics", None)
    legacy = importlib.import_module("tests.legacy_metrics")
    _drop_metrics_modules()

    importlib.import_module("trend_analysis.metrics")

    assert sys.modules["tests.legacy_metrics"] is legacy
    assert isinstance(legacy, ModuleType)
    assert legacy.__file__
    assert Path(legacy.__file__).parts[-2:] == ("tests", "legacy_metrics.py")


def test_metrics_first_import_does_not_shadow_legacy_metrics_module() -> None:
    sys.modules.pop("tests.legacy_metrics", None)
    _drop_metrics_modules()

    importlib.import_module("trend_analysis.metrics")
    legacy = importlib.import_module("tests.legacy_metrics")

    assert sys.modules["tests.legacy_metrics"] is legacy
    assert isinstance(legacy, ModuleType)
    assert legacy.__file__
    assert Path(legacy.__file__).parts[-2:] == ("tests", "legacy_metrics.py")
