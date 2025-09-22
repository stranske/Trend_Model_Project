from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest


def test_fallback_loader_activates_when_imports_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "trend_analysis"
        / "config"
        / "models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "trend_analysis.config.models_fallback_loader_test", module_path
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "trend_analysis.config.model" or (level == 1 and name == "model"):
            raise ImportError("forced failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    spec.loader.exec_module(module)

    assert module.validate_trend_config.__module__ == module.__name__

    validated = module.validate_trend_config({}, base_path=Path.cwd())
    assert isinstance(validated, dict)


def test_fallback_loader_returns_minimal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "trend_analysis"
        / "config"
        / "models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "trend_analysis.config.models_fallback_payload_test", module_path
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "trend_analysis.config.model" or (level == 1 and name == "model"):
            raise ImportError("forced failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    spec.loader.exec_module(module)

    payload = {
        "data": {"csv_path": "demo.csv"},
        "portfolio": {"max_turnover": 0.1},
        "vol_adjust": {"target_vol": 1.0},
        "extra": {"ignored": True},
    }

    validated = module.validate_trend_config(payload, base_path=Path.cwd())

    assert set(validated.keys()) >= {"data", "portfolio", "vol_adjust"}
    assert "extra" not in validated
