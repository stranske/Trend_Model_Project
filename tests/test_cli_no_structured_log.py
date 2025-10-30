import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def test_cli_no_structured_log(tmp_path: Path):
    # Prepare minimal returns CSV
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-31", periods=8, freq="ME"),
            "Fund_A": 0.01,
            "Fund_B": 0.012,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {
            "in_start": "2023-01",
            "in_end": "2023-04",
            "out_start": "2023-05",
            "out_end": "2023-08",
        },
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "transaction_cost_bps": 10,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    log_path = tmp_path / "explicit.jsonl"
    # Run CLI with --no-structured-log and explicit log file path
    cmd = [
        sys.executable,
        "-m",
        "trend_analysis.cli",
        "run",
        "-c",
        str(cfg_path),
        "-i",
        str(csv_path),
        "--log-file",
        str(log_path),
        "--no-structured-log",
    ]
    env = {
        **dict(**os.environ),
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
    }
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=env,
    )
    assert proc.returncode == 0
    # Log file should not exist because logging disabled
    assert not log_path.exists()
