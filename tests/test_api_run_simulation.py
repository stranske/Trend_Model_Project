import pandas as pd
from trend_analysis.config import Config
from trend_analysis import api, pipeline


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def make_cfg(path: str | None = None) -> Config:
    cfg = Config(
        version="1",
        data={"csv_path": path} if path else {},
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


def test_run_simulation_matches_pipeline(tmp_path):
    df = make_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = make_cfg(str(csv))

    expected_details = pipeline.run_full(cfg)
    expected_metrics = pipeline.run(cfg)

    result = api.run_simulation(cfg, df)

    assert result.details["benchmark_ir"] == expected_details["benchmark_ir"]
    assert result.details["out_sample_stats"] == expected_details["out_sample_stats"]
    pd.testing.assert_frame_equal(
        result.details["score_frame"], expected_details["score_frame"]
    )
    pd.testing.assert_frame_equal(result.metrics, expected_metrics)
