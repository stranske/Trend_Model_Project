import sys
import pathlib
import yaml  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402


def test_load_defaults():
    cfg = config.load()
    with open(config.DEFAULTS, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    assert cfg.version == data.get("version")
    assert "data" in cfg.model_dump()
