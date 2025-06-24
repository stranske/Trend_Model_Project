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

