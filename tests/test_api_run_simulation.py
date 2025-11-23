import hashlib
import json

import pandas as pd

from trend_analysis import api, pipeline
from trend_analysis.config import Config


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def make_cfg(path: str | None = None) -> Config:
    data_section: dict[str, object] = {
        "risk_free_column": "RF",
        "allow_risk_free_fallback": False,
        "date_column": "Date",
        "frequency": "M",
    }
    if path:
        data_section["csv_path"] = path
    cfg = Config(
        version="1",
        data=data_section,
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
            if arr.dtype.kind in "fc":  # float or complex
                arr = np.round(arr, 12)  # Use more precision for better stability
            # Sort the array if it's 1D for determinism
            if arr.ndim == 1:
                arr = np.sort(arr)
            return arr.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Use to_dict with records orientation and sort for complete determinism
            df_copy = obj.copy()
            # Round floating point columns to avoid precision issues
            float_cols = df_copy.select_dtypes(
                include=[np.float64, np.float32, np.float16]
            ).columns
            for col in float_cols:
                df_copy[col] = df_copy[col].round(12)
            # Sort by index and columns for determinism
            df_copy = df_copy.sort_index().sort_index(axis=1)
            # Use a more deterministic serialization
            result = {}
            for col in sorted(df_copy.columns):
                result[col] = df_copy[col].tolist()
            return result
        elif isinstance(obj, pd.Series):
            series_copy = obj.copy()
            if series_copy.dtype.kind in "fc":  # float or complex
                series_copy = series_copy.round(12)
            # Sort series by index for determinism
            series_copy = series_copy.sort_index()
            return series_copy.to_dict()
        elif isinstance(obj, (float, np.floating)):
            # Round floating point numbers to avoid precision issues
            return round(float(obj), 12)
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples recursively and sort if they contain comparable items
            try:
                processed = [deterministic_default(item) for item in obj]
                # Only sort if all items are of compatible types
                if processed and all(
                    type(item) is type(processed[0]) for item in processed
                ):
                    processed = sorted(processed)
                return processed
            except (TypeError, AttributeError):
                return [deterministic_default(item) for item in obj]
        elif isinstance(obj, dict):
            # Sort dictionary by keys for determinism
            return {k: deterministic_default(v) for k, v in sorted(obj.items())}
        else:
            return str(obj)

    # Ensure metrics DataFrame is sorted for determinism
    metrics_copy = res.metrics.copy().sort_index().sort_index(axis=1)

    # Create more stable serialization
    # Prefer sanitized details view if provided by api.run_simulation
    details_obj = getattr(res, "details_sanitized", res.details)
    payload = {
        "metrics": deterministic_default(metrics_copy),
        "details": deterministic_default(details_obj),
        "seed": res.seed,
        "environment": deterministic_default(res.environment),
    }

    # Use deterministic JSON serialization
    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def test_run_simulation_deterministic(tmp_path):
    df = make_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = make_cfg(str(csv))
    cfg.seed = 123

    # Import required modules
    import random

    import numpy as np

    def reset_random_state(seed):
        """Reset all random state to ensure deterministic behavior."""
        # Set Python's built-in random seed
        random.seed(seed)
        # Set NumPy's global random seed
        np.random.seed(seed)
        # Also reset the global random state to ensure clean slate
        random.getstate()  # This forces initialization of the random state
        np.random.get_state()  # This forces initialization of numpy's state

    # Run 1: Set seeds and run simulation
    reset_random_state(cfg.seed)
    r1 = api.run_simulation(cfg, df)

    # Run 2: Reset seeds and run simulation again
    reset_random_state(cfg.seed)
    r2 = api.run_simulation(cfg, df)

    # Generate hashes
    hash1 = _hash_result(r1)
    hash2 = _hash_result(r2)

    # If hashes don't match, provide detailed debugging info
    if hash1 != hash2:
        print(f"Hash mismatch - Run 1: {hash1}, Run 2: {hash2}")
        print(f"Seeds - Run 1: {r1.seed}, Run 2: {r2.seed}")
        print(f"Environment - Run 1 Python: {r1.environment.get('python')}")
        print(f"Environment - Run 2 Python: {r2.environment.get('python')}")
        print(f"Metrics equal: {r1.metrics.equals(r2.metrics)}")
        if not r1.metrics.equals(r2.metrics):
            print("Metrics diff:")
            try:
                print(r1.metrics.compare(r2.metrics))
            except Exception as e:
                print(f"Could not compare metrics: {e}")
                print("Run 1 metrics shape:", r1.metrics.shape)
                print("Run 2 metrics shape:", r2.metrics.shape)
                print("Run 1 metrics columns:", list(r1.metrics.columns))
                print("Run 2 metrics columns:", list(r2.metrics.columns))
                print("Run 1 metrics dtypes:", r1.metrics.dtypes.to_dict())
                print("Run 2 metrics dtypes:", r2.metrics.dtypes.to_dict())

        print(f"Details keys - Run 1: {sorted(r1.details.keys())}")
        print(f"Details keys - Run 2: {sorted(r2.details.keys())}")

        # Check specific detail items for differences
        for key in sorted(r1.details.keys()):
            if key in r2.details:
                val1 = r1.details[key]
                val2 = r2.details[key]
                if isinstance(val1, dict) and isinstance(val2, dict):
                    if val1.keys() != val2.keys():
                        print(
                            f"Different keys in details[{key}]: "
                            f"{val1.keys()} vs {val2.keys()}"
                        )
                elif val1 != val2:
                    print(
                        f"Different values in details[{key}]: "
                        f"{type(val1)} vs {type(val2)}"
                    )
            else:
                print(f"Key {key} missing in Run 2 details")

        # Check for keys only in Run 2
        for key in sorted(r2.details.keys()):
            if key not in r1.details:
                print(f"Key {key} only in Run 2 details")

    assert hash1 == hash2, f"Results are not deterministic: {hash1} != {hash2}"


def test_run_simulation_deterministic_with_random_selection(tmp_path):
    """Test determinism specifically with random portfolio selection mode."""
    # Create a larger dataset to trigger random selection
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    data = {"Date": dates, "RF": 0.0}
    # Add many funds to trigger random selection
    for i in range(15):
        data[f"Fund_{i:02d}"] = [0.01 + (i * 0.001) + (j * 0.0001) for j in range(12)]

    df = pd.DataFrame(data)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    cfg = make_cfg(str(csv))
    cfg.seed = 456  # Different seed to test
    # Configure for random selection with limited funds
    cfg.portfolio = {"selection_mode": "random", "random_n": 5}
    cfg.sample_split = {
        "in_start": "2020-01",
        "in_end": "2020-06",
        "out_start": "2020-07",
        "out_end": "2020-12",
    }

    import random

    import numpy as np

    def reset_random_state(seed):
        """Reset all random state to ensure deterministic behavior."""
        random.seed(seed)
        np.random.seed(seed)
        random.getstate()  # Force initialization
        np.random.get_state()  # Force initialization

    # Run 1: Set seeds and run simulation
    reset_random_state(cfg.seed)
    r1 = api.run_simulation(cfg, df)

    # Run 2: Reset seeds and run simulation again
    reset_random_state(cfg.seed)
    r2 = api.run_simulation(cfg, df)

    # Generate hashes
    hash1 = _hash_result(r1)
    hash2 = _hash_result(r2)

    # Check that the same funds were selected
    funds1 = r1.details.get("selected_funds", [])
    funds2 = r2.details.get("selected_funds", [])
    assert funds1 == funds2, f"Different funds selected: {funds1} vs {funds2}"
    assert len(funds1) == 5, f"Expected 5 funds, got {len(funds1)}"

    # If hashes don't match, provide debugging info
    if hash1 != hash2:
        print(
            f"Hash mismatch in random selection test - Run 1: {hash1}, Run 2: {hash2}"
        )
        print(f"Selected funds - Run 1: {funds1}")
        print(f"Selected funds - Run 2: {funds2}")
        print(f"Seeds - Run 1: {r1.seed}, Run 2: {r2.seed}")

    assert (
        hash1 == hash2
    ), f"Results are not deterministic with random selection: {hash1} != {hash2}"
