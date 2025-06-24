import sys
import pathlib
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import config


def test_load_defaults():
    cfg = config.load()
    with open(config.DEFAULTS, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    assert cfg.version == data.get("version")
    assert "data" in cfg.model_dump()
