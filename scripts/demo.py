from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import shutil
from typing import Any, Dict

import nbformat

import pandas as pd
import yaml

from trend_analysis.config import Config, load as load_config
from trend_analysis import (
    pipeline,
    export,
    run_analysis,
    cli,
    data,
    metrics,
)
from trend_analysis.multi_period.scheduler import generate_periods
from trend_analysis.multi_period.engine import run as run_multi
from trend_analysis.multi_period.replacer import Rebalancer
from tools import strip_output


@export.register_formatter_excel("metrics")
def _fmt_metrics(ws, wb) -> None:
    """Simple metrics sheet formatting for the demo."""
    ws.freeze_panes(1, 0)


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


def main(out_dir: str | Path | None = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    csv = root / "hedge_fund_returns_with_indexes.csv"
    cfg = make_config(csv)

    # demonstrate the lightweight CLI
    with NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        yaml.safe_dump(cfg.model_dump(), fh)
        cfg_file = fh.name
    cli_rc = cli.main(["-c", cfg_file, "--version"])
    cli_json_rc = cli.main(["-c", cfg_file])
    detailed_rc = run_analysis.main(["-c", cfg_file, "--detailed"])
    os.environ["TREND_CFG"] = cfg_file
    loaded_cfg = load_config()
    os.environ.pop("TREND_CFG", None)

    # run via the pipeline and CLI entry point
    run_rc = run_analysis.main(["-c", cfg_file])

    df = data.load_csv(str(csv))
    if df is None:
        df = pd.DataFrame()
    rf_col = data.identify_risk_free_fund(df)
    df = data.ensure_datetime(df, "Date")
    available = metrics.available_metrics()

    score_frame = pipeline.single_period_run(
        df, cfg.sample_split["in_start"], cfg.sample_split["in_end"]
    )
    full_res = pipeline.run_full(cfg)
    metrics_df = pipeline.run(cfg)
    rb = Rebalancer({})
    init_wt = pd.Series(1 / len(score_frame.columns), index=score_frame.columns)
    rb_weights = rb.apply_triggers(init_wt, score_frame)

    # multi-period components
    cfg_dict = cfg.model_dump()
    if hasattr(cfg, "multi_period"):
        cfg_dict["multi_period"] = getattr(cfg, "multi_period")
    periods = generate_periods(cfg_dict)
    mp_res = run_multi(cfg_dict)

    out_dir_path = Path(out_dir) if out_dir else root / "demo_outputs"
    out_dir_path.mkdir(exist_ok=True)

    nb_src = root / "Vol_Adj_Trend_Analysis1.5.TrEx.ipynb"
    nb_copy = out_dir_path / "tmp_nb.ipynb"
    shutil.copy(nb_src, nb_copy)
    strip_output.strip_output(str(nb_copy))
    nb = nbformat.read(nb_copy, as_version=nbformat.NO_CONVERT)
    nb_clean = all(not c.get("outputs") for c in nb.cells)

    sheet_fmt = export.make_summary_formatter(
        full_res,
        cfg.sample_split["in_start"],
        cfg.sample_split["in_end"],
        cfg.sample_split["out_start"],
        cfg.sample_split["out_end"],
    )
    text_summary = export.format_summary_text(
        full_res,
        cfg.sample_split["in_start"],
        cfg.sample_split["in_end"],
        cfg.sample_split["out_start"],
        cfg.sample_split["out_end"],
    )

    frames = {"metrics": metrics_df, "summary": pd.DataFrame()}
    export.export_to_excel(
        frames, str(out_dir_path / "analysis.xlsx"), default_sheet_formatter=sheet_fmt
    )
    export.export_data(frames, str(out_dir_path / "analysis"), formats=["csv", "json"])

    print(text_summary)
    print("Risk-free column:", rf_col)
    print("Available metrics:", available)
    print(metrics_df.head())
    print(score_frame.head())
    os.remove(cfg_file)

    return {
        "cli_rc": cli_rc,
        "cli_json_rc": cli_json_rc,
        "run_rc": run_rc,
        "detailed_rc": detailed_rc,
        "score_frame": score_frame,
        "full_res": full_res,
        "metrics_df": metrics_df,
        "periods": periods,
        "mp_res": mp_res,
        "out_dir": out_dir_path,
        "rf_col": rf_col,
        "summary_text": text_summary,
        "available": available,
        "rb_weights": rb_weights.to_dict(),
        "loaded_version": loaded_cfg.version,
        "nb_clean": nb_clean,
    }


if __name__ == "__main__":
    main()
