from trend_analysis import config

def test_load_defaults():
    cfg = config.load()
    assert cfg.version
    assert "data" in cfg.model_dump()
