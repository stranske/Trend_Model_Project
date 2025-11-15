from __future__ import annotations

import json
from pathlib import Path

from trend import cli


def _write_artifacts(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / f"metrics_{run_id}.csv"
    metrics_path.write_text(
        "index,Sharpe\nFundA,0.5\n",
        encoding="utf-8",
    )
    details = {
        "portfolio_equal_weight_combined": {"2022-01-31": 0.01},
        "parameter_grid": {"values": [[0.1, 0.2], [0.3, 0.4]]},
    }
    (run_dir / f"details_{run_id}.json").write_text(
        json.dumps(details),
        encoding="utf-8",
    )
    (run_dir / f"summary_{run_id}.txt").write_text(
        "Quick summary text",
        encoding="utf-8",
    )
    return run_dir


def test_trend_quick_report_subcommand_creates_report(tmp_path: Path) -> None:
    base_dir = tmp_path / "perf"
    run_id = "demo123"
    artifacts = _write_artifacts(base_dir, run_id)
    config_path = tmp_path / "config.yml"
    config_path.write_text("sample: config", encoding="utf-8")

    exit_code = cli.main(
        [
            "quick-report",
            "--run-id",
            run_id,
            "--artifacts",
            str(artifacts),
            "--base-dir",
            str(base_dir),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    html_path = base_dir / "reports" / f"{run_id}.html"
    heatmap_path = base_dir / f"heatmap_{run_id}.png"
    assert html_path.exists()
    assert heatmap_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "sample: config" in html
    assert "Quick summary text" in html
