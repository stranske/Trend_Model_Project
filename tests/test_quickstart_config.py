from __future__ import annotations

from pathlib import Path

from trend_analysis.config import load_trend_config


def test_quickstart_config_loads() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "quickstart.yml"
    cfg, resolved = load_trend_config(cfg_path)

    assert resolved.name == "quickstart.yml"
    assert cfg.data.csv_path.exists()
    assert cfg.data.date_column == "Date"
