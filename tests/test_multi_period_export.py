import pandas as pd
import yaml
from pathlib import Path

from trend_analysis.config import Config
from trend_analysis.multi_period import run as run_mp
from trend_analysis.export import metrics_from_result, export_multi_period_metrics


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


def test_export_multi_period_metrics(tmp_path):
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    out = tmp_path / "res"
    export_multi_period_metrics(results, str(out), formats=["csv"])
    first_period = results[0]["period"][3]
    second_period = results[1]["period"][3]
    p1 = out.with_name(f"{out.stem}_{first_period}.csv")
    p2 = out.with_name(f"{out.stem}_{second_period}.csv")
    assert p1.exists() and p2.exists()
    df_read = pd.read_csv(p1)
    assert "cagr" in df_read.columns
