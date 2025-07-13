from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import pandas as pd
import yaml

from trend_analysis.config import Config
from trend_analysis import pipeline, export, run_analysis, cli
from trend_analysis.multi_period.scheduler import generate_periods
from trend_analysis.multi_period.engine import run as run_multi


def make_config(csv: Path) -> Config:
    return Config(
        version="1",
        data={"csv_path": str(csv)},
        preprocessing={},
        vol_adjust={"target_vol": 0.1},
        sample_split={
            "in_start": "2012-01",
            "in_end": "2012-06",
            "out_start": "2012-07",
            "out_end": "2012-12",
        },
        portfolio={
            "selection_mode": "rank",
            "rank": {"inclusion_approach": "top_n", "n": 3, "score_by": "AnnualReturn"},
        },
        metrics={},
        export={},
        run={},
        multi_period={
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2012-01",
            "end": "2012-12",
        },
    )


def main(out_dir: str | Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    csv = root / "hedge_fund_returns_with_indexes.csv"
    cfg = make_config(csv)

    # demonstrate the lightweight CLI
    with NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        yaml.safe_dump(cfg.model_dump(), fh)
        cfg_file = fh.name
    cli.main(["-c", cfg_file, "--version"])

    # run via the pipeline and CLI entry point
    run_analysis.main(["-c", cfg_file])

    df = pd.read_csv(csv)
    score_frame = pipeline.single_period_run(
        df, cfg.sample_split["in_start"], cfg.sample_split["in_end"]
    )
    full_res = pipeline.run_full(cfg)
    metrics_df = pipeline.run(cfg)

    # multi-period components
    generate_periods(cfg.model_dump())
    run_multi(cfg.model_dump())

    out_dir_path = Path(out_dir) if out_dir else root / "demo_outputs"
    out_dir_path.mkdir(exist_ok=True)

    sheet_fmt = export.make_summary_formatter(
        full_res,
        cfg.sample_split["in_start"],
        cfg.sample_split["in_end"],
        cfg.sample_split["out_start"],
        cfg.sample_split["out_end"],
    )

    data = {"metrics": metrics_df, "summary": pd.DataFrame()}
    export.export_to_excel(
        data, str(out_dir_path / "analysis.xlsx"), default_sheet_formatter=sheet_fmt
    )
    export.export_data(data, str(out_dir_path / "analysis"), formats=["csv", "json"])

    print(metrics_df.head())
    print(score_frame.head())
    os.remove(cfg_file)


if __name__ == "__main__":
    main()
