import pandas as pd
import json
import hashlib

from trend_analysis.config import Config
from trend_analysis import api, pipeline


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
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
    assert result.seed == cfg.seed
    assert "python" in result.environment


def _hash_result(res: api.RunResult) -> str:
    def deterministic_default(obj):
        import datetime
        import numpy as np
        import pandas as pd

        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Use to_dict with a fixed orientation for determinism
            return obj.to_dict(orient="list")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return str(obj)

    payload = {
        "metrics": res.metrics.to_json(),
        "details": json.dumps(
            res.details, sort_keys=True, default=deterministic_default
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=deterministic_default).encode()
    ).hexdigest()


def test_run_simulation_deterministic(tmp_path):
    df = make_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = make_cfg(str(csv))
    cfg.seed = 123

    r1 = api.run_simulation(cfg, df)
    r2 = api.run_simulation(cfg, df)

    assert _hash_result(r1) == _hash_result(r2)
