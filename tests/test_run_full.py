import pandas as pd
import pytest

from trend_analysis.config import Config
from trend_analysis.diagnostics import PipelineResult
from trend_analysis.pipeline import run, run_full


def make_cfg(tmp_path, df):
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = Config(
        version="1",
        data={
            "csv_path": str(csv),
            "date_column": "Date",
            "frequency": "M",
            "risk_free_column": "RF",
        },
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
    assert isinstance(detailed, PipelineResult)
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


def test_run_full_with_benchmarks(tmp_path):
    df = make_df()
    df["SPX"] = 0.02
    cfg = make_cfg(tmp_path, df)
    cfg.benchmarks = {"spx": "SPX"}
    res = run_full(cfg)
    assert "spx" in res["benchmark_ir"]
    assert "equal_weight" in res["benchmark_ir"]["spx"]
    assert "A" in res["benchmark_ir"]["spx"]


def test_run_full_respects_metric_registry(tmp_path):
    df = make_df()
    cfg = make_cfg(tmp_path, df)
    cfg.metrics = {"registry": ["sharpe_ratio", "volatility"]}
    res = run_full(cfg)
    sf = res["score_frame"]
    assert sf.columns.tolist() == ["Sharpe", "Volatility"]
