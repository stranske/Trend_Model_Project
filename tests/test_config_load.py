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
