from pathlib import Path

import pandas as pd

from trend_analysis import cli


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


def test_cli_version_custom(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, "1.2.3")
    csv = tmp_path / "data.csv"
    csv.write_text("Date,RF\n2020-01-31,0.0\n")

    captured: dict[str, str] = {}

    def fake_run(cfg):
        captured["version"] = cfg.version
        return pd.DataFrame()

    monkeypatch.setattr(cli.pipeline, "run", fake_run)
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: {"dummy": 1})
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    assert rc == 0
    assert captured["version"] == "1.2.3"


def test_cli_default_json(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, "1")
    csv = tmp_path / "data.csv"
    csv.write_text("Date,RF\n2020-01-31,0.0\n")

    monkeypatch.setattr(cli.pipeline, "run", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: None)

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "No results"
