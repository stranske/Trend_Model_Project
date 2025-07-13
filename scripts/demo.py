from __future__ import annotations

from pathlib import Path
import pandas as pd

from trend_analysis.config import Config
from trend_analysis import pipeline, export


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
            "rank": {"inclusion_approach": "top_n", "n": 3},
        },
        metrics={},
        export={},
        run={},
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv = root / "hedge_fund_returns_with_indexes.csv"
    cfg = make_config(csv)

    full_res = pipeline.run_full(cfg)
    metrics_df = pipeline.run(cfg)

    out_dir = root / "demo_outputs"
    out_dir.mkdir(exist_ok=True)

    sheet_fmt = export.make_summary_formatter(
        full_res,
        cfg.sample_split["in_start"],
        cfg.sample_split["in_end"],
        cfg.sample_split["out_start"],
        cfg.sample_split["out_end"],
    )

    data = {"metrics": metrics_df, "summary": pd.DataFrame()}
    export.export_to_excel(
        data, str(out_dir / "analysis.xlsx"), default_sheet_formatter=sheet_fmt
    )
    export.export_data(data, str(out_dir / "analysis"), formats=["csv", "json"])

    print(metrics_df.head())
    print(full_res["score_frame"].head())


if __name__ == "__main__":
    main()
