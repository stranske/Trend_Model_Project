import pandas as pd
import numpy as np

from trend_analysis.multi_period import engine


def make_df() -> pd.DataFrame:
    dates = pd.date_range("2019-12-31", periods=60, freq="ME")
    data = {"Date": dates}
    for i in range(5):
        data[f"F{i}"] = 0.01 * (i + 1)
    return pd.DataFrame(data)


def make_cfg(df: pd.DataFrame, tmp_path) -> dict:
    return {
        "dataframe": df,
        "data": {"csv_path": ""},
        "vol_adjust": {"target_vol": 1.0},
        "run": {"monthly_cost": 0.0},
        "portfolio": {},
        "multi_period": {
            "frequency": "A",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020",
            "end": "2023",
            "triggers": {"sigma1": {"sigma": 1, "periods": 1}},
            "min_funds": 2,
            "max_funds": 5,
            "weight_curve": {"anchors": [[0, 1.0], [100, 1.0]]},
        },
        "checkpoint_dir": str(tmp_path / "chk"),
        "random_seed": 0,
    }


def fake_single_period_run(*args, **kwargs):
    df = args[0]
    funds = [c for c in df.columns if c != "Date"]
    score = pd.DataFrame(
        {
            "zscore": np.linspace(-1.5, 1.5, len(funds)),
            "rank": np.arange(1, len(funds) + 1),
        },
        index=funds,
    )
    return {"score_frame": score, "selected_funds": funds}


def test_engine_run_summary(tmp_path, monkeypatch):
    df = make_df()
    cfg = make_cfg(df, tmp_path)
    monkeypatch.setattr(engine, "single_period_run", fake_single_period_run)
    result = engine.run(cfg)
    assert isinstance(result["summary"], pd.DataFrame)
    assert len(result["summary"]) == 3
