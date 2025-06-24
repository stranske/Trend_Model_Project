import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import config

def test_load_defaults():
    cfg = config.load()
    assert cfg.version
    assert "data" in cfg.model_dump()
