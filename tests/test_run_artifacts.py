from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trend_analysis.reporting.run_artifacts import write_run_artifacts


def test_write_run_artifacts_copies_files_and_builds_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-31", periods=3, freq="ME"),
            "Mgr_A": [0.1, 0.2, -0.05],
        }
    )
    metrics = pd.DataFrame({"cagr": [0.12, 0.08], "sharpe": [1.1, 0.9]})
    stats = SimpleNamespace(
        cagr=0.15,
        vol=0.04,
        sharpe=1.2,
        sortino=0.9,
        max_drawdown=0.11,
        information_ratio=0.85,
    )
    details = {"out_ew_stats": stats, "selected_funds": ["Mgr_A", "Mgr_B"]}

    export_dir = tmp_path / "outputs"
    export_dir.mkdir()
    artifact = export_dir / "analysis.xlsx"
    artifact.write_text("demo", encoding="utf-8")

    input_path = tmp_path / "input.csv"
    input_path.write_text("Date,M1\n2021-01-31,0.01\n", encoding="utf-8")

    run_dir = write_run_artifacts(
        output_dir=export_dir,
        run_id="abc123def456",
        config={"alpha": 1},
        config_path="config.yml",
        input_path=input_path,
        data_frame=df,
        metrics_frame=metrics,
        run_details=details,
        exported_files=[artifact],
        summary_text="Summary block",
    )

    manifest_path = run_dir / "manifest.json"
    html_path = run_dir / "report.html"
    copied_artifact = run_dir / "analysis.xlsx"
    assert manifest_path.exists()
    assert html_path.exists()
    assert copied_artifact.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "abc123def456"
    assert manifest["artifacts"] and manifest["artifacts"][0]["name"] == "analysis.xlsx"
    assert manifest["data_window"]["instrument_count"] == 1
    assert manifest["config_snapshot"]["alpha"] == 1
    assert manifest["html_report"] == "report.html"
    assert manifest["selected_funds"] == ["Mgr_A", "Mgr_B"]

    html = html_path.read_text(encoding="utf-8")
    assert "Trend run receipt" in html
    assert "Summary block" in html


def test_write_run_artifacts_skips_missing_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"Date": pd.date_range("2022-01-31", periods=1, freq="ME")})
    run_dir = write_run_artifacts(
        output_dir=tmp_path / "out",
        run_id="no-files",
        config={},
        config_path="cfg.yml",
        input_path=tmp_path / "missing.csv",
        data_frame=df,
        metrics_frame=pd.DataFrame(),
        run_details={},
        exported_files=[tmp_path / "out" / "missing.xlsx"],
        summary_text="",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == []
    assert manifest["data_window"]["rows"] == 1
