import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import config

def test_load_defaults():
    cfg = config.load()
    assert cfg.version
    assert "data" in cfg.model_dump()

import yaml

def test_load_custom(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    yaml.dump(
        {
            "version": "x",
            "data": {},
            "preprocessing": {},
            "vol_adjust": {},
            "sample_split": {},
            "portfolio": {},
            "metrics": {},
            "export": {},
            "run": {},
        },
        open(cfg_path, "w"),
    )
    cfg = config.load(str(cfg_path))
    assert cfg.version == "x"

