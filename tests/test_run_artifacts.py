from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trend_analysis.reporting import run_artifacts
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


def test_serialise_stats_handles_mapping_and_invalid_values() -> None:
    stats = {"cagr": "0.5", "vol": "n/a", "ignored": "x"}

    result = run_artifacts._serialise_stats(stats)

    assert result == {"cagr": 0.5}
    assert run_artifacts._serialise_stats(None) == {}


def test_data_window_uses_index_when_date_column_missing() -> None:
    df = pd.DataFrame(
        {"asset1": [1, 2], "asset2": [3, 4]},
        index=pd.period_range("2020-01", periods=2, freq="M"),
    )

    summary = run_artifacts._data_window(df)

    assert summary["rows"] == 2
    assert summary["instrument_count"] == 2
    assert summary["start"].startswith("2020-01")
    assert summary["end"].startswith("2020-02")


def test_render_html_renders_metrics_and_artifacts() -> None:
    created = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)
    manifest = {
        "metrics": {"cagr": 0.1234},
        "artifacts": [{"name": "demo.csv", "size": 4}],
        "data_window": {"start": "2020-01-01", "end": "2020-02-01", "instrument_count": 1},
        "git_hash": "deadbeef",
    }

    html = run_artifacts._render_html(
        run_id="rid", created=created, manifest=manifest, summary_text="hello"
    )

    assert "Trend run receipt" in html
    assert "0.1234" in html
    assert "demo.csv" in html
    assert "hello" in html


def test_write_run_artifacts_deduplicates_exports_and_coerces_selected(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(run_artifacts, "_git_hash", lambda: "fixedhash")

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")

    run_dir = write_run_artifacts(
        output_dir=tmp_path / "out",
        run_id="dup-test",
        config=None,
        config_path="cfg.yml",
        input_path=tmp_path / "missing.csv",
        data_frame=[{"A": 1}],
        metrics_frame=[{"cagr": 0.2, "sharpe": 1.1}],
        run_details={"selected_funds": "Alpha"},
        exported_files=[artifact, artifact],
        summary_text="",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["selected_funds"] == ["Alpha"]
    assert len(manifest["artifacts"]) == 1
    assert manifest["metrics_overview"]["avg_cagr"] == 0.2


def test_coerce_frame_gracefully_handles_unexpected_input() -> None:
    class Bad:
        pass

    result = run_artifacts._coerce_frame(Bad())

    assert isinstance(result, pd.DataFrame)
    assert result.empty
