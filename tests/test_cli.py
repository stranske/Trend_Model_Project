import pathlib
import os
import sys
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis import cli, config


def make_cfg_file(path: pathlib.Path) -> pathlib.Path:
    base = config.load()
    data = base.model_dump()
    data["data"]["returns_path"] = "tests/data/sample.csv"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return path


def test_cli_main_env(tmp_path, monkeypatch, capsys):
    cfg_file = make_cfg_file(tmp_path / "cfg.yml")
    monkeypatch.setenv("TREND_CFG", str(cfg_file))
    cli.main([])
    out = capsys.readouterr().out
    assert "selected_funds" in out


def test_cli_main_arg(tmp_path, capsys):
    cfg_file = make_cfg_file(tmp_path / "cfg.yml")
    cli.main([str(cfg_file)])
    out = capsys.readouterr().out
    assert "selected_funds" in out
