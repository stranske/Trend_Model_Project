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
            # Sort arrays for determinism and handle floating point precision
            arr = np.array(obj)
            if arr.dtype.kind in 'fc':  # float or complex
                arr = np.round(arr, 10)  # Round to 10 decimals for float precision
            return arr.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Use to_dict with records orientation and sort for complete determinism
            df_copy = obj.copy()
            # Round floating point columns to avoid precision issues
            float_cols = df_copy.select_dtypes(include=[np.float64, np.float32]).columns
            for col in float_cols:
                df_copy[col] = df_copy[col].round(10)
            # Sort by index and columns for determinism
            df_copy = df_copy.sort_index().sort_index(axis=1)
            return df_copy.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            series_copy = obj.copy()
            if series_copy.dtype.kind in 'fc':  # float or complex
                series_copy = series_copy.round(10)
            # Sort series by index for determinism
            series_copy = series_copy.sort_index()
            return series_copy.to_dict()
        elif isinstance(obj, (float, np.floating)):
            # Round floating point numbers to avoid precision issues
            return round(float(obj), 10)
        else:
            return str(obj)

    # Ensure metrics DataFrame is sorted for determinism
    metrics_copy = res.metrics.copy().sort_index().sort_index(axis=1)
    
    payload = {
        "metrics": metrics_copy.to_json(orient="records", date_format="iso"),
        "details": json.dumps(res.details, sort_keys=True, default=deterministic_default),
        "seed": res.seed,  # Include seed for additional verification
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=deterministic_default).encode()).hexdigest()


def test_run_simulation_deterministic(tmp_path):
    df = make_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = make_cfg(str(csv))
    cfg.seed = 123
    
    # Ensure clean state before each run
    import os
    import random
    import numpy as np
    
    # Run 1: Set seeds and run simulation
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    r1 = api.run_simulation(cfg, df)
    
    # Run 2: Reset seeds and run simulation again
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    r2 = api.run_simulation(cfg, df)
    
    # Generate hashes
    hash1 = _hash_result(r1)
    hash2 = _hash_result(r2)
    
    # If hashes don't match, provide debugging info
    if hash1 != hash2:
        print(f"Hash mismatch - Run 1: {hash1}, Run 2: {hash2}")
        print(f"Seeds - Run 1: {r1.seed}, Run 2: {r2.seed}")
        print(f"Environment - Run 1 Python: {r1.environment.get('python')}")
        print(f"Environment - Run 2 Python: {r2.environment.get('python')}")
        print(f"Metrics equal: {r1.metrics.equals(r2.metrics)}")
        if not r1.metrics.equals(r2.metrics):
            print("Metrics diff:")
            print(r1.metrics.compare(r2.metrics))
        print(f"Details keys - Run 1: {sorted(r1.details.keys())}")
        print(f"Details keys - Run 2: {sorted(r2.details.keys())}")
    
    assert hash1 == hash2, f"Results are not deterministic: {hash1} != {hash2}"
