from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from trend_analysis.config import Config
from trend_analysis.export import (
    combined_summary_frame,
    combined_summary_result,
    export_multi_period_metrics,
    export_phase1_multi_metrics,
    export_phase1_workbook,
    flat_frames_from_results,
    metrics_from_result,
    period_frames_from_results,
    phase1_workbook_data,
    summary_frame_from_result,
    workbook_frames_from_results,
)
from trend_analysis.multi_period import run as run_mp


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


def _period_label(res_obj) -> str:
    return str(cast(dict, res_obj)["period"][3])


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


def test_combined_summary_frame(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frame = combined_summary_frame(results)
    other = summary_frame_from_result(combined_summary_result(results))
    pd.testing.assert_frame_equal(frame, other)


def test_period_frames_from_results_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frames = period_frames_from_results(results)
    assert len(frames) == len(results)
    key = _period_label(results[0])
    assert key in frames
    assert "OS MaxDD" in frames[key].columns


def test_workbook_frames_from_results_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frames = workbook_frames_from_results(results)
    first = _period_label(results[0])
    assert "summary" in frames
    assert "execution_metrics" in frames
    assert first in frames
    assert list(frames[first].columns) == list(frames["summary"].columns)


def test_phase1_workbook_data(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frames = phase1_workbook_data(results, include_metrics=True)
    first = _period_label(results[0])
    assert f"metrics_{first}" in frames
    assert "summary" in frames
    assert "execution_metrics" in frames


def test_flat_frames_from_results_basic():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    frames = flat_frames_from_results(results)
    # Must at least contain periods, summary, and execution metrics; may also include 'changes'
    assert {"periods", "summary", "execution_metrics"}.issubset(set(frames))
    assert not frames["periods"].empty
    cols_periods = [c for c in frames["periods"].columns if c != "Period"]
    assert cols_periods == list(frames["summary"].columns)


def test_export_multi_period_metrics(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_multi_period_metrics(results, str(out), formats=["csv"])
    periods_path = out.with_name(f"{out.stem}_periods.csv")
    summ = out.with_name(f"{out.stem}_summary.csv")
    assert periods_path.exists() and summ.exists()
    df_read = pd.read_csv(periods_path)
    assert list(df_read.columns)[0] == "Period"
    assert df_read.iloc[0, 1] == "Equal Weight"


def test_export_multi_period_metrics_excel(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_multi_period_metrics(results, str(out), formats=["xlsx"])
    path = out.with_suffix(".xlsx")
    assert path.exists()
    first_period = _period_label(results[0])
    second_period = _period_label(results[1])
    book = pd.ExcelFile(path)
    assert first_period in book.sheet_names
    assert second_period in book.sheet_names
    assert "summary" in book.sheet_names


def test_export_phase1_multi_metrics(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_phase1_multi_metrics(results, str(out), formats=["csv"])
    periods_path = out.with_name(f"{out.stem}_periods.csv")
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    exec_path = out.with_name(f"{out.stem}_execution_metrics.csv")
    assert periods_path.exists() and summary_path.exists() and exec_path.exists()
    files = set(tmp_path.glob("*.csv"))
    assert {periods_path, summary_path, exec_path}.issubset(files)
    df_read = pd.read_csv(periods_path)
    assert list(df_read.columns)[0] == "Period"
    assert df_read.iloc[0, 1] == "Equal Weight"


def test_export_phase1_multi_metrics_json(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_phase1_multi_metrics(results, str(out), formats=["json"])
    periods_path = out.with_name(f"{out.stem}_periods.json")
    summary_path = out.with_name(f"{out.stem}_summary.json")
    exec_path = out.with_name(f"{out.stem}_execution_metrics.json")
    assert periods_path.exists() and summary_path.exists() and exec_path.exists()
    files = set(tmp_path.glob("*.json"))
    assert {periods_path, summary_path, exec_path}.issubset(files)
    df_read = pd.read_json(periods_path)
    assert list(df_read.columns)[0] == "Period"
    assert df_read.loc[0, "Name"] == "Equal Weight"


def test_export_phase1_workbook(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res.xlsx"
    export_phase1_workbook(results, str(out))
    assert out.exists()
    first_period = _period_label(results[0])
    second_period = _period_label(results[1])
    book = pd.ExcelFile(out)
    assert first_period in book.sheet_names
    assert second_period in book.sheet_names
    assert "summary" in book.sheet_names
    assert "execution_metrics" in book.sheet_names


def test_export_phase1_workbook_content(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res.xlsx"
    export_phase1_workbook(results, str(out))
    book = pd.read_excel(out, sheet_name=None, skiprows=4)
    first_period = _period_label(results[0])
    df_first = book[first_period]
    df_summary = book["summary"]
    df_exec = pd.read_excel(out, sheet_name="execution_metrics")
    assert list(df_first.columns) == list(df_summary.columns)
    assert df_summary.iloc[0, 0] == "Equal Weight"
    assert list(df_exec.columns) == ["Period", "Turnover", "Transaction Cost"]


def test_export_phase1_workbook_order(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res.xlsx"
    export_phase1_workbook(results, str(out))
    book = pd.ExcelFile(out)
    expected = ["summary", "execution_metrics"] + [
        str(cast(dict, r)["period"][3]) for r in results
    ]
    assert book.sheet_names == expected


def test_export_phase1_workbook_metrics(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res.xlsx"
    export_phase1_workbook(results, str(out), include_metrics=True)
    book = pd.ExcelFile(out)
    first_period = _period_label(results[0])
    assert f"metrics_{first_period}" in book.sheet_names
    assert "metrics_summary" in book.sheet_names


def test_export_phase1_multi_metrics_excel(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_phase1_multi_metrics(results, str(out), formats=["xlsx"])
    path = out.with_suffix(".xlsx")
    assert path.exists()
    first_period = _period_label(results[0])
    second_period = _period_label(results[1])
    book = pd.ExcelFile(path)
    assert first_period in book.sheet_names
    assert second_period in book.sheet_names
    assert "summary" in book.sheet_names
    assert "execution_metrics" in book.sheet_names
    df_first = pd.read_excel(path, sheet_name=first_period, skiprows=4)
    df_summary = pd.read_excel(path, sheet_name="summary", skiprows=4)
    assert list(df_first.columns) == list(df_summary.columns)
