import json
from pathlib import Path
import pytest

from trend_analysis import cli
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


def test_cli_version_custom(tmp_path, capsys):
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, "1.2.3")
    rc = cli.main(["--version", "-c", str(cfg)])
    captured = capsys.readouterr().out.strip()
    assert rc == 0
    assert captured == "1.2.3"


def test_cli_default_json(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    loaded = json.loads(out)
    assert rc == 0
    assert loaded["version"] == config.load().version

