import pytest

from trend_analysis.config import Config


def _minimal_valid_config(**overrides):
    base = {
        "version": "1.0",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "run": {},
    }
    base.update(overrides)
    return base


def test_sections_require_mappings():
    cfg = _minimal_valid_config(data=0)
    with pytest.raises(TypeError, match="data must be a dictionary"):
        Config(**cfg)


def test_version_must_be_string():
    cfg = _minimal_valid_config(version=0)
    with pytest.raises(TypeError, match="version must be a string"):
        Config(**cfg)
