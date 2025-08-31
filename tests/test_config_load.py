# Unit tests for config.load
from pathlib import Path

from trend_analysis import config


def _write_cfg(path: Path, version: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"version: '{version}'",
                "data: {}",
                "preprocessing: {}",
                "vol_adjust: {}",
                "sample_split: {}",
                "portfolio: {}",
                "metrics: {}",
                "export: {}",
                "run: {}",
            ]
        )
    )


def test_load_default():
    cfg = config.load()
    assert isinstance(cfg, config.Config)
    assert cfg.version


def test_load_custom(tmp_path):
    path = tmp_path / "c.yml"
    _write_cfg(path, "99")
    cfg = config.load(str(path))
    assert cfg.version == "99"


def test_env_var_override(tmp_path, monkeypatch):
    cfg_file = tmp_path / "env.yml"
    _write_cfg(cfg_file, "42")
    monkeypatch.setenv("TREND_CFG", str(cfg_file))
    cfg = config.load()
    assert cfg.version == "42"
    monkeypatch.delenv("TREND_CFG", raising=False)


def test_output_alias(tmp_path):
    cfg_file = tmp_path / "alias.yml"
    cfg_file.write_text(
        "\n".join(
            [
                "version: '1'",
                "data: {}",
                "preprocessing: {}",
                "vol_adjust: {}",
                "sample_split: {}",
                "portfolio: {}",
                "metrics: {}",
                "output: {format: csv, path: '" + str(tmp_path / "res") + "'}",
                "run: {}",
            ]
        )
    )
    cfg = config.load(str(cfg_file))
    assert cfg.export["formats"] == ["csv"]
    assert cfg.export["directory"] == str(tmp_path)
    assert cfg.export["filename"] == "res"


def test_empty_version_rejected(tmp_path):
    """Test that empty version field is rejected with proper validation."""
    cfg_file = tmp_path / "empty_version.yml"
    cfg_file.write_text(
        "\n".join(
            [
                'version: ""',
                "data: {}",
                "preprocessing: {}",
                "vol_adjust: {}",
                "sample_split: {}",
                "portfolio: {}",
                "metrics: {}",
                "export: {}",
                "run: {}",
            ]
        )
    )
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "String should have at least 1 character" in str(exc_info.value)


def test_whitespace_version_rejected(tmp_path):
    """Test that whitespace-only version field is rejected with clear error."""
    cfg_file = tmp_path / "whitespace_version.yml"
    cfg_file.write_text(
        "\n".join(
            [
                'version: "   "',
                "data: {}",
                "preprocessing: {}",
                "vol_adjust: {}",
                "sample_split: {}",
                "portfolio: {}",
                "metrics: {}",
                "export: {}",
                "run: {}",
            ]
        )
    )
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "Version field cannot be empty" in str(exc_info.value)
