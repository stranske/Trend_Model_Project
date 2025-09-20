from pathlib import Path
import types

import pandas as pd
import pytest

from trend_analysis import run_analysis


def _write_cfg(path: Path, csv: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                f"data: {{csv_path: '{csv}'}}",
                "preprocessing: {}",
                "vol_adjust: {target_vol: 1.0}",
                "sample_split: {in_start: '2020-01', in_end: '2020-03', "
                "out_start: '2020-04', out_end: '2020-06'}",
                "portfolio: {}",
                "metrics: {}",
                "export: {}",
                "run: {}",
            ]
        )
    )


def _make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_cli_detailed(tmp_path, capsys):
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, csv)
    rc = run_analysis.main(["-c", str(cfg), "--detailed"])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "cagr" in captured.lower()


def test_main_raises_when_csv_path_missing(monkeypatch):
    cfg = types.SimpleNamespace(
        data={},
        sample_split={},
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    with pytest.raises(KeyError):
        run_analysis.main(["-c", "dummy.yml"])


def test_main_raises_when_csv_missing(monkeypatch):
    cfg = types.SimpleNamespace(
        data={"csv_path": "missing.csv"},
        sample_split={},
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda path: None)

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["-c", "dummy.yml"])
