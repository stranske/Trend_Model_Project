import pandas as pd
import pytest

from trend_analysis.pipeline import run_full, run
from trend_analysis.config import Config


def make_cfg(tmp_path, df):
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = Config(
        version="1",
        data={"csv_path": str(csv)},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        portfolio={},
        metrics={},
        export={},
        run={},
    )
    return cfg


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_run_full_matches_run(tmp_path):
    cfg = make_cfg(tmp_path, make_df())
    detailed = run_full(cfg)
    summary = run(cfg)
    assert isinstance(detailed, dict)
    assert not summary.empty
    stats = detailed["out_sample_stats"]
    assert set(summary.index) == set(stats.keys())

def test_run_full_missing_csv_key(tmp_path):
    cfg = Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={},
        sample_split={},
        portfolio={},
        metrics={},
        export={},
        run={},
    )
    with pytest.raises(KeyError):
        run_full(cfg)


def test_run_full_missing_file(tmp_path):
    cfg = Config(
        version="1",
        data={"csv_path": str(tmp_path / "missing.csv")},
        preprocessing={},
        vol_adjust={},
        sample_split={},
        portfolio={},
        metrics={},
        export={},
        run={},
    )
    with pytest.raises(FileNotFoundError):
        run_full(cfg)
