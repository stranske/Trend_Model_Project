from pathlib import Path

import pandas as pd

from trend_analysis import cli
from trend_analysis.api import RunResult


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

    def fake_run(cfg, _df):
        captured["version"] = cfg.version
        return RunResult(
            metrics=pd.DataFrame(),
            details={"out_sample_stats": {}},
            seed=42,
            environment={},
            summary_text="",
        )

    monkeypatch.setattr(cli, "run_simulation", fake_run)
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

    monkeypatch.setattr(
        cli,
        "run_simulation",
        lambda cfg, _df: RunResult(
            metrics=pd.DataFrame(),
            details={},
            seed=42,
            environment={},
            summary_text=None,
        ),
    )

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "No results"
