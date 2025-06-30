import pytest
from trend_analysis import config


def test_load_non_mapping(tmp_path):
    cfg_file = tmp_path / "invalid.yml"
    cfg_file.write_text("- a\n- b\n")
    with pytest.raises(TypeError):
        config.load(cfg_file)
