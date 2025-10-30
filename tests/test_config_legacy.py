from pathlib import Path

import pytest
import yaml

from trend_analysis.config import legacy


def sample_config(tmp_path: Path) -> Path:
    data = {
        "version": "1",
        "data": {
            "managers_glob": "data/raw/managers/*.csv",
            "date_column": "Date",
            "frequency": "D",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0,
        },
        "metrics": {},
        "export": {},
        "run": {},
        "output": {"format": "csv", "path": str(tmp_path / "out" / "results.csv")},
    }
    cfg_path = tmp_path / "config.yml"
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return cfg_path


def test_load_uses_output_section(tmp_path: Path):
    cfg_path = sample_config(tmp_path)
    cfg = legacy.load(cfg_path)
    assert cfg.export["formats"] == ["csv"]
    assert cfg.export["directory"].endswith("out")
    assert cfg.export["filename"] == "results.csv"


def test_load_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = sample_config(tmp_path)
    monkeypatch.setenv("TREND_CFG", str(cfg_path))
    cfg = legacy.load()
    assert cfg.version == "1"


def test_load_non_mapping(tmp_path: Path):
    bad_path = tmp_path / "bad.yml"
    bad_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TypeError):
        legacy.load(bad_path)
