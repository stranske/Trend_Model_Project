from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.cli as cli
from analysis.cv import walk_forward


def _synth_returns(length: int = 12) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=length, freq="ME")
    data = {
        "asset_a": np.linspace(0.01, 0.06, length),
        "asset_b": np.linspace(0.02, 0.07, length),
    }
    return pd.DataFrame(data, index=index)


def test_walk_forward_expanding_boundaries():
    data = _synth_returns()
    report = walk_forward(data, folds=3, expand=True, params={"top_n": 1})

    folds = report.folds
    assert list(folds["fold"]) == [1, 2, 3]
    assert folds.loc[0, "train_end"] == data.index[2]
    assert folds.loc[0, "test_start"] == data.index[3]
    assert folds.loc[1, "train_end"] == data.index[5]
    assert folds.loc[1, "test_end"] == data.index[8]
    assert folds.loc[2, "test_end"] == data.index[-1]


def test_walk_forward_avoids_lookahead():
    index = pd.date_range("2021-01-01", periods=12, freq="ME")
    data = pd.DataFrame(
        {
            "asset_a": [
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
            ],
            "asset_b": [
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                0.3,
                0.3,
                0.3,
                0.3,
                0.3,
                0.3,
            ],
        },
        index=index,
    )
    report = walk_forward(data, folds=3, expand=True, params={"top_n": 1, "seed": 0})
    selected = list(report.folds["selected"])

    assert selected[0] == "asset_a"
    assert selected[1] == "asset_a"
    assert selected[2] == "asset_b"


def test_cli_generates_outputs(tmp_path: Path):
    csv_path = tmp_path / "returns.csv"
    df = _synth_returns(length=10)
    df = df.reset_index().rename(columns={"index": "Date"})
    df.to_csv(csv_path, index=False)

    cfg_path = tmp_path / "cv.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                f"  csv_path: {csv_path}",
                "  date_column: Date",
                "  columns: [asset_a, asset_b]",
                "cv:",
                "  folds: 2",
                "  expand: true",
                "params:",
                "  top_n: 1",
                "  lookback: 5",
                "  cost_per_turnover: 0.0",
                "output:",
                f"  dir: {tmp_path/'out'}",
            ]
        )
    )

    exit_code = cli.main(
        [
            "cv",
            "--config",
            str(cfg_path),
            "--folds",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0

    folds_path = tmp_path / "out" / "cv_folds.csv"
    summary_path = tmp_path / "out" / "cv_summary.csv"
    markdown_path = tmp_path / "out" / "cv_report.md"

    assert folds_path.exists()
    assert summary_path.exists()
    assert markdown_path.exists()

    folds_df = pd.read_csv(folds_path)
    assert {"oos_sharpe", "turnover", "cost_drag"}.issubset(folds_df.columns)

    summary_df = pd.read_csv(summary_path)
    assert summary_df.loc[0, "folds"] == 2
