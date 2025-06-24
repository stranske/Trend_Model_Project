import pathlib
import os
import sys
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import config


def test_load_defaults():
    cfg = config.load()
    assert cfg.version
    assert "data" in cfg.model_dump()


def test_load_env_override(tmp_path, monkeypatch):
    base = config.load()
    data = base.model_dump()
    data["data"]["returns_path"] = "tests/data/sample.csv"
    cfg_path = tmp_path / "cfg.yml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    monkeypatch.setenv("TREND_CFG", str(cfg_path))
    cfg = config.load(None)
    assert cfg.data["returns_path"].endswith("sample.csv")
