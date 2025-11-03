from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from trend_analysis.cli import _apply_trend_spec_preset
from trend_analysis.config import load_config
from trend_analysis.signal_presets import get_trend_spec_preset


@pytest.fixture()
def base_config() -> object:
    sample_csv = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "managers"
        / "sample_manager.csv"
    )
    cfg_dict = {
        "version": "0.1",
        "data": {
            "date_column": "Date",
            "frequency": "M",
            "csv_path": str(sample_csv),
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }
    return load_config(cfg_dict)


def test_apply_trend_spec_preset_sets_signals(base_config: object) -> None:
    preset = get_trend_spec_preset("Balanced")
    _apply_trend_spec_preset(base_config, preset)
    signals = getattr(base_config, "signals")
    assert signals["window"] == preset.spec.window
    assert signals["lag"] == preset.spec.lag
    assert signals["vol_adjust"] is preset.spec.vol_adjust
    assert pytest.approx(signals.get("vol_target", 0.0), rel=1e-6) == pytest.approx(
        preset.spec.vol_target or 0.0
    )
    assert getattr(base_config, "trend_spec_preset") == "Balanced"


def test_apply_trend_spec_preset_on_mapping() -> None:
    preset = get_trend_spec_preset("Conservative")
    cfg: dict[str, object] = {}
    _apply_trend_spec_preset(cfg, preset)
    assert cfg["trend_spec_preset"] == "Conservative"
    assert cfg["signals"]["window"] == preset.spec.window


@dataclass(init=False)
class FrozenConfig:
    signals: dict[str, object]
    trend_spec_preset: str | None = None

    def __init__(
        self, signals: dict[str, object], trend_spec_preset: str | None = None
    ) -> None:
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "trend_spec_preset", trend_spec_preset)

    def __setattr__(self, name: str, value: object) -> None:
        raise ValueError("frozen")


def test_apply_trend_spec_preset_uses_object_setattr() -> None:
    preset = get_trend_spec_preset("Aggressive")
    cfg = FrozenConfig(signals={})
    _apply_trend_spec_preset(cfg, preset)
    assert cfg.signals["window"] == preset.spec.window
    assert cfg.trend_spec_preset == "Aggressive"
