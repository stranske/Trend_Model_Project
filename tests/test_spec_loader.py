from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from trend_model import spec as spec_module
from trend_model.spec import (
    TrendRunSpec,
    ensure_run_spec,
    load_run_spec_from_file,
    load_run_spec_from_mapping,
)


@dataclass(frozen=True)
class _FrozenConfig:
    signals: dict[str, Any]
    sample_split: dict[str, str]
    portfolio: dict[str, Any]


def test_load_run_spec_from_file() -> None:
    cfg_path = Path("config") / "trend.toml"
    spec = load_run_spec_from_file(cfg_path)

    assert spec.trend.window > 0
    assert spec.trend.lag >= 1
    assert spec.backtest.selection_mode == "rank"
    assert "annual_return" in spec.backtest.metrics
    assert spec.backtest.export_formats


def test_load_run_spec_from_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError):
        load_run_spec_from_file(missing_path)

    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text("ignored = true", encoding="utf-8")

    monkeypatch.setattr(spec_module.tomllib, "load", lambda fh: [1, 2, 3])

    with pytest.raises(TypeError):
        load_run_spec_from_file(invalid_path)


def test_ensure_run_spec_attaches_attributes() -> None:
    cfg_path = Path("config") / "trend.toml"
    spec = load_run_spec_from_file(cfg_path)
    cfg = spec.config
    # Simulate loader attaching to a clone of the config object
    if hasattr(cfg, "model_dump"):
        try:
            clone = cfg.__class__(**cfg.model_dump())
        except TypeError as e:
            raise RuntimeError(
                f"Failed to clone config object: {e}. "
                "Ensure that model_dump() returns keys compatible with the constructor."
            )
    else:
        clone = cfg
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


def test_ensure_run_spec_handles_failures() -> None:
    class Explosive:
        def __getattr__(self, name: str) -> Any:  # pragma: no cover - fallback guard
            raise RuntimeError("boom")

    assert ensure_run_spec(Explosive()) is None


def test_ensure_run_spec_uses_object_setattr() -> None:
    cfg = _FrozenConfig(
        signals={"window": 10, "lag": 1, "vol_adjust": True, "vol_target": 0.2},
        sample_split={"in_start": "2020-01", "out_start": "2021-01"},
        portfolio={},
    )

    ensured = ensure_run_spec(cfg)

    assert ensured is not None
    assert getattr(cfg, "trend_spec").vol_target == 0.2


def test_helper_coercion_and_mapping_behaviour(tmp_path: Path) -> None:
    class WithModelDump:
        def model_dump(self) -> dict[str, Any]:
            return {"a": 1}

    class WithBadModelDump:
        def model_dump(self) -> int:
            return 5

    class WithDict:
        def __init__(self) -> None:
            self.b = 2

    assert spec_module._as_mapping({"key": "value"}) == {"key": "value"}
    assert spec_module._as_mapping(WithModelDump()) == {"a": 1}
    assert spec_module._as_mapping(WithBadModelDump()) == {}
    assert spec_module._as_mapping(WithDict()) == {"b": 2}
    assert spec_module._as_mapping(object()) == {}

    assert spec_module._coerce_float(None) is None
    assert spec_module._coerce_float("", default=1.2) == 1.2
    assert spec_module._coerce_float("3.5") == 3.5
    assert spec_module._coerce_float("bad", default=7.1) == 7.1

    assert spec_module._as_tuple([1, 2, 3]) == (1, 2, 3)
    assert spec_module._as_tuple(("x", "y")) == ("x", "y")
    assert spec_module._as_tuple([1, 2], coerce=str) == ("1", "2")

    base = tmp_path / "base"
    base.mkdir()
    rel = spec_module._maybe_path("relative.txt", base_path=base)
    assert rel == (base / "relative.txt").resolve()
    assert spec_module._maybe_path("", base_path=base) is None
    absolute = spec_module._maybe_path(str(base / "abs.txt"), base_path=base)
    assert absolute == (base / "abs.txt").resolve()


def test_build_trend_and_backtest_specs(tmp_path: Path) -> None:
    payload = {
        "version": "1",
        "data": {
            "csv_path": "returns.csv",
            "date_column": "Date",
            "frequency": "ME",
        },
        "signals": {
            "window": 5,
            "lag": 2,
            "vol_adjust": False,
            "vol_target": 0.1,
            "min_periods": -5,
        },
        "vol_adjust": {"target_vol": 0.15, "floor_vol": 0.05, "warmup_periods": 0},
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        "portfolio": {
            "selection_mode": "manual",
            "rebalance_calendar": "NYSE",
            "transaction_cost_bps": 10,
            "max_turnover": 0.5,
            "manual_list": ["A", "B"],
            "indices_list": ("IDX",),
            "rank": {"foo": "bar"},
            "selector": {"name": "rank"},
            "weighting": {"name": "equal"},
            "custom_weights": {"A": 0.7},
        },
        "export": {"directory": "exports", "formats": ["csv"]},
        "output": {"path": "report.html", "format": "csv"},
        "run": {"seed": 7, "monthly_cost": "0.05"},
        "metrics": {"registry": ("sharpe",)},
        "benchmarks": {"spx": "SPX"},
        "regime": {"enabled": True},
    }

    base = tmp_path / "cfg"
    base.mkdir()
    (base / "returns.csv").write_text("date,value\n2020-01,1\n", encoding="utf-8")

    spec = load_run_spec_from_mapping(payload, base_path=base)

    assert spec.trend.vol_target is None  # Disabled by vol_adjust False
    assert spec.trend.min_periods is None
    assert spec.backtest.manual_list == ("A", "B")
    assert spec.backtest.indices_list == ("IDX",)
    assert spec.backtest.export_directory == (base / "exports").resolve()
    assert spec.backtest.output_path == (base / "report.html").resolve()
    assert spec.backtest.export_formats == ("csv",)


def test_temporary_cwd_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    original = Path.cwd()
    try:
        with spec_module._temporary_cwd(None):
            assert Path.cwd() == original
        with spec_module._temporary_cwd(target):
            assert Path.cwd() == target
    finally:
        assert Path.cwd() == original
