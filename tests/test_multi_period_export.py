import pandas as pd
import yaml
from pathlib import Path

from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_mp
from trend_analysis.export import (
    metrics_from_result,
    summary_frame_from_result,
    combined_summary_result,
    export_multi_period_metrics,
    period_frames_from_results,
)


def make_df():
    dates = pd.date_range("1990-01-31", periods=12, freq="ME")
    return pd.DataFrame({"Date": dates, "A": 0.01, "B": 0.02})


def make_cfg():
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-04",
    }
    return Config(**cfg_data)


def test_metrics_from_result_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    df_metrics = metrics_from_result(results[0])
    assert set(df_metrics.columns).issuperset(
        {
            "cagr",
            "vol",
            "sharpe",
            "sortino",
            "information_ratio",
            "max_drawdown",
        }
    )
    assert not df_metrics.empty


def test_summary_frame_from_result_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    df_sum = summary_frame_from_result(results[0])
    assert "OS MaxDD" in df_sum.columns
    assert df_sum.iloc[0, 0] == "Equal Weight"


def test_combined_summary_result_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    summary = combined_summary_result(results)
    df_sum = summary_frame_from_result(summary)
    assert "OS MaxDD" in df_sum.columns
    assert df_sum.iloc[0, 0] == "Equal Weight"


def test_period_frames_from_results_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frames = period_frames_from_results(results)
    assert len(frames) == len(results)
    key = str(results[0]["period"][3])
    assert key in frames
    assert "OS MaxDD" in frames[key].columns


def test_export_multi_period_metrics(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_multi_period_metrics(results, str(out), formats=["csv"])
    periods_path = out.with_name(f"{out.stem}_periods.csv")
    summ = out.with_name(f"{out.stem}_summary.csv")
    assert periods_path.exists() and summ.exists()
    df_read = pd.read_csv(periods_path, index_col=0)
    assert "Name" in df_read.columns
    assert df_read.iloc[0, 0] == "Equal Weight"


def test_export_multi_period_metrics_excel(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_multi_period_metrics(results, str(out), formats=["xlsx"])
    path = out.with_suffix(".xlsx")
    assert path.exists()
    first_period = str(results[0]["period"][3])
    second_period = str(results[1]["period"][3])
    book = pd.ExcelFile(path)
    assert first_period in book.sheet_names
    assert second_period in book.sheet_names
    assert "summary" in book.sheet_names
    df_read = pd.read_excel(path, sheet_name=first_period, header=None)
    assert "Vol-Adj Trend Analysis" in df_read.iloc[0, 0]
