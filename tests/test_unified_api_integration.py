"""Comprehensive test validating unified API behavior across CLI and
components."""

import tempfile
from pathlib import Path

import pandas as pd

from trend_analysis import api
from trend_analysis.config import Config
from trend_analysis.data import load_csv


def test_comprehensive_api_integration():
    """Test that all components use the unified API correctly."""
    # Create test data
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    test_df = pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "Manager_A": [0.02 + 0.001 * i for i in range(12)],
            "Manager_B": [0.015 + 0.002 * i for i in range(12)],
            "SPX": [0.01 + 0.001 * i for i in range(12)],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_data.csv"
        test_df.to_csv(csv_file, index=False)

        # Create config
        config = Config(
            version="1",
            data={"csv_path": str(csv_file)},
            preprocessing={},
            vol_adjust={"target_vol": 1.0},
            sample_split={
                "in_start": "2020-01",
                "in_end": "2020-06",
                "out_start": "2020-07",
                "out_end": "2020-12",
            },
            portfolio={},
            benchmarks={"spx": "SPX"},
            metrics={},
            export={},
            run={},
            seed=42,
        )

        # Test 1: API direct call
        result = api.run_simulation(config, test_df)

        # Validate RunResult structure
        assert hasattr(result, "metrics"), "RunResult missing metrics"
        assert hasattr(result, "details"), "RunResult missing details"
        assert hasattr(result, "seed"), "RunResult missing seed"
        assert hasattr(result, "environment"), "RunResult missing environment"

        # Validate metrics DataFrame
        assert isinstance(result.metrics, pd.DataFrame), "metrics should be DataFrame"
        expected_columns = {
            "cagr",
            "vol",
            "sharpe",
            "sortino",
            "information_ratio",
            "max_drawdown",
        }
        actual_columns = set(result.metrics.columns)
        assert expected_columns.issubset(
            actual_columns
        ), f"Missing columns: {expected_columns - actual_columns}"

        # Validate details dictionary
        assert isinstance(result.details, dict), "details should be dict"
        assert "out_sample_stats" in result.details, "details missing out_sample_stats"
        assert "benchmark_ir" in result.details, "details missing benchmark_ir"

        # Validate seed
        assert result.seed == 42, "seed should match config"

        # Validate environment
        assert isinstance(result.environment, dict), "environment should be dict"
        assert "python" in result.environment, "environment missing python version"
        assert "numpy" in result.environment, "environment missing numpy version"
        assert "pandas" in result.environment, "environment missing pandas version"

        # Test 2: Reproducibility
        result2 = api.run_simulation(config, test_df)
        pd.testing.assert_frame_equal(
            result.metrics, result2.metrics, "API calls should be reproducible"
        )

        # Test 3: CLI integration works by loading CSV internally
        # (CLI now loads CSV and calls api.run_simulation internally)
        # This validates the unified code path
        loaded_df = load_csv(str(csv_file))
        assert loaded_df is not None, "CSV loading should work"
        expected_df = test_df.copy()
        expected_df["Date"] = expected_df["Date"].dt.tz_localize("UTC")
        pd.testing.assert_frame_equal(
            expected_df, loaded_df, "Loaded CSV should match original"
        )

        # Validate that CLI would get same data
        cli_result = api.run_simulation(config, loaded_df)
        pd.testing.assert_frame_equal(
            result.metrics, cli_result.metrics, "CLI path should match API call"
        )

        print("✓ All integration tests passed!")
        print(f"✓ API processed {len(result.metrics)} funds")
        print(f"✓ Generated metrics with {len(result.metrics.columns)} columns")
        print(f"✓ Environment: Python {result.environment['python']}")


if __name__ == "__main__":
    test_comprehensive_api_integration()
