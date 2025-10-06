from __future__ import annotations

import os
from pathlib import Path

from trend_model.spec import TrendRunSpec, ensure_run_spec, load_run_spec_from_file


def test_load_run_spec_from_file() -> None:
    cfg_path = Path("config") / "trend.toml"
    spec = load_run_spec_from_file(cfg_path)

    assert spec.trend.window > 0
    assert spec.trend.lag >= 1
    assert spec.backtest.selection_mode == "rank"
    assert "annual_return" in spec.backtest.metrics
    assert spec.backtest.export_formats


def test_ensure_run_spec_attaches_attributes() -> None:
    cfg_path = Path("config") / "trend.toml"
    spec = load_run_spec_from_file(cfg_path)
    cfg = spec.config
    # Simulate loader attaching to a clone of the config object
    clone = cfg.__class__(**cfg.model_dump()) if hasattr(cfg, "model_dump") else cfg
    original_cwd = Path.cwd()
    try:
        os.chdir(cfg_path.parent)
        ensured = ensure_run_spec(clone, base_path=cfg_path.parent)
    finally:
        os.chdir(original_cwd)

    assert ensured is not None
    attached = getattr(clone, "_trend_run_spec", None)
    assert isinstance(attached, TrendRunSpec)
    assert hasattr(clone, "trend_spec")
    assert hasattr(clone, "backtest_spec")
    assert clone.trend_spec.window == ensured.trend.window
    assert ensured.backtest.selection_mode == "rank"
