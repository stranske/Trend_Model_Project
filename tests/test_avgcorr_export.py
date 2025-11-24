from pathlib import Path

import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.export import summary_frame_from_result
from trend_analysis.pipeline import _run_analysis  # type: ignore

RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def _make_dataset(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    f1_in = np.array([0.01, 0.02, 0.03])
    f1_out = np.array([0.015, 0.025, 0.035])
    f1 = np.concatenate([f1_in, f1_out])
    f2 = f1.copy()  # perfectly correlated with f1 (corr=1)
    f3 = -f1  # perfectly negatively correlated with f1 and f2 (corr=-1)
    rf = np.zeros_like(f1)
    df = pd.DataFrame(
        {
            "Date": dates,
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "RF": rf,
        }
    )
    csv_path = tmp_path / "avgcorr_demo.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_avgcorr_present_when_requested(tmp_path):
    csv_path = _make_dataset(tmp_path)
    stats_cfg = RiskStatsConfig(
        metrics_to_run=[
            "AnnualReturn",
            "Volatility",
            "Sharpe",
            "Sortino",
            "MaxDrawdown",
            "InformationRatio",
            "AvgCorr",
        ],
        risk_free=0.0,
    )
    full = pd.read_csv(csv_path)
    res = _run_analysis(
        full,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        target_vol=0.1,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=stats_cfg,
        **RUN_KWARGS,
    )
    assert res is not None
    frame = summary_frame_from_result(res)
    assert "IS AvgCorr" in frame.columns
    assert "OS AvgCorr" in frame.columns
    f1_row = frame[frame["Name"] == "F1"].iloc[0]
    f2_row = frame[frame["Name"] == "F2"].iloc[0]
    f3_row = frame[frame["Name"] == "F3"].iloc[0]
    tol = 1e-12
    assert abs(f1_row["IS AvgCorr"] - 0.0) <= tol
    assert abs(f2_row["IS AvgCorr"] - 0.0) <= tol
    assert abs(f3_row["IS AvgCorr"] - (-1.0)) <= tol
    assert abs(f1_row["OS AvgCorr"] - 0.0) <= tol
    assert abs(f2_row["OS AvgCorr"] - 0.0) <= tol
    assert abs(f3_row["OS AvgCorr"] - (-1.0)) <= tol


def test_avgcorr_absent_when_not_requested(tmp_path):
    csv_path = _make_dataset(tmp_path)
    stats_cfg = RiskStatsConfig(
        metrics_to_run=[
            "AnnualReturn",
            "Volatility",
            "Sharpe",
            "Sortino",
            "MaxDrawdown",
            "InformationRatio",
        ],
        risk_free=0.0,
    )
    full = pd.read_csv(csv_path)
    res = _run_analysis(
        full,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        target_vol=0.1,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=stats_cfg,
        **RUN_KWARGS,
    )
    assert res is not None
    frame = summary_frame_from_result(res)
    assert "IS AvgCorr" not in frame.columns
    assert "OS AvgCorr" not in frame.columns
