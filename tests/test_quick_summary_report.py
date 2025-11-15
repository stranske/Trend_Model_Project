from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trend.reporting.quick_summary import build_run_report, main


def _write_run_artifacts(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        {"Sharpe": [0.82, 0.74], "CAGR": [0.11, 0.09]}, index=["FundA", "FundB"]
    )
    metrics.to_csv(run_dir / f"metrics_{run_id}.csv")

    summary_text = "Run complete\nOut-of-sample delivered 11% CAGR."
    (run_dir / f"summary_{run_id}.txt").write_text(summary_text, encoding="utf-8")

    details = {
        "portfolio_equal_weight_combined": {
            "2021-01-31": 0.012,
            "2021-02-28": -0.004,
            "2021-03-31": 0.010,
        },
        "risk_diagnostics": {
            "turnover": {
                "2021-01-31": 0.18,
                "2021-02-28": 0.22,
                "2021-03-31": 0.20,
            }
        },
        "parameter_grid": {
            "rows": ["window_12", "window_24"],
            "columns": ["lag_1", "lag_2"],
            "values": [[0.12, 0.14], [0.15, 0.18]],
        },
    }
    (run_dir / f"details_{run_id}.json").write_text(
        json.dumps(details), encoding="utf-8"
    )
    return run_dir


def test_build_run_report_creates_html_and_heatmap(tmp_path: Path) -> None:
    base_dir = tmp_path / "perf"
    run_id = "demo123"
    run_dir = _write_run_artifacts(base_dir, run_id)
    config_path = tmp_path / "config.yml"
    config_path.write_text("sample: config\nvalue: 1\n", encoding="utf-8")

    output = build_run_report(
        run_id,
        run_dir,
        config_text=config_path.read_text(encoding="utf-8"),
        base_dir=base_dir,
    )

    html = output.read_text(encoding="utf-8")
    assert "Trend run report" in html
    assert "sample: config" in html
    assert "data:image/png;base64" in html
    assert f"heatmap_{run_id}.png" in html

    heatmap_path = base_dir / f"heatmap_{run_id}.png"
    assert heatmap_path.exists()


def test_cli_infers_run_id_and_handles_missing_sections(tmp_path: Path) -> None:
    base_dir = tmp_path / "perf"
    run_id = "auto-id"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame({"Sharpe": [0.5]}, index=["FundA"])
    metrics.to_csv(run_dir / f"metrics_{run_id}.csv")
    details = {"portfolio_equal_weight_combined": {"2022-01-31": 0.01}}
    (run_dir / f"details_{run_id}.json").write_text(
        json.dumps(details), encoding="utf-8"
    )

    exit_code = main(["--artifacts", str(run_dir), "--base-dir", str(base_dir)])
    assert exit_code == 0

    html_path = base_dir / "reports" / f"{run_id}.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "No turnover data available." in html

    heatmap_path = base_dir / f"heatmap_{run_id}.png"
    assert heatmap_path.exists()
