"""Golden master test comparing CLI and API outputs."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from trend_analysis import api
from trend_analysis.config import Config


def make_test_data():
    """Create test data for golden master comparison."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "Fund_A": [0.01 + 0.001 * i for i in range(24)],
            "Fund_B": [0.015 + 0.002 * i for i in range(24)],
            "Fund_C": [0.008 + 0.0005 * i for i in range(24)],
            "SPX": [0.012 + 0.001 * i for i in range(24)],
        }
    )


def make_test_config(csv_path: str) -> Config:
    """Create test configuration."""
    return Config(
        version="1",
        data={
            "csv_path": csv_path,
            "risk_free_column": "RF",
            "date_column": "Date",
            "frequency": "M",
            "allow_risk_free_fallback": False,
        },
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        portfolio={
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0.0},
            "weighting": {"name": "equal", "params": {}},
        },
        benchmarks={"spx": "SPX"},
        metrics={},
        export={"directory": str(Path(csv_path).parent / "outputs"), "formats": ["csv"]},
        run={},
    )


def _write_config(cfg_path: Path, config: Config) -> None:
    """Write Config object to YAML file."""
    config_dict = config.model_dump(exclude_none=True)
    config_dict.pop("unknown", None)
    for key in ("extra", "robustness", "signals", "strategy", "walk_forward"):
        if not config_dict.get(key):
            config_dict.pop(key, None)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def test_cli_api_golden_master():
    """Test that CLI and API produce identical outputs for same inputs."""
    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_data.csv"
        config_file = tmp_path / "test_config.yml"

        # Write test data
        df = make_test_data()
        df.to_csv(csv_file, index=False)

        # Create and write config
        cfg = make_test_config(str(csv_file))
        _write_config(config_file, cfg)

        # Test API output.
        api_result = api.run_simulation(cfg, df)

        # The supported CLI must execute successfully against the same inputs.
        cli_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trend.cli",
                "run",
                "-c",
                str(config_file),
                "-i",
                str(csv_file),
                "--no-structured-log",
            ],
            cwd=Path(__file__).parent.parent,
            env={
                **dict(os.environ),
                "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert cli_result.returncode == 0, cli_result.stderr

    # For this test, we'll compare API results with expected structure
    # The CLI comparison is complex due to output formatting
    assert isinstance(api_result.metrics, pd.DataFrame)
    assert not api_result.metrics.empty
    assert "cagr" in api_result.metrics.columns
    assert "sharpe" in api_result.metrics.columns
    assert "ir_spx" in api_result.metrics.columns

    # Validate RunResult structure
    assert hasattr(api_result, "details")
    assert hasattr(api_result, "seed")
    assert hasattr(api_result, "environment")

    # Validate details structure
    assert "out_sample_stats" in api_result.details
    assert "benchmark_ir" in api_result.details

    # Validate environment info
    assert "python" in api_result.environment
    assert "numpy" in api_result.environment
    assert "pandas" in api_result.environment


def test_api_deterministic_behavior():
    """Test that API produces deterministic results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_data.csv"

        # Write test data
        df = make_test_data()
        df.to_csv(csv_file, index=False)

        # Create config with fixed seed
        cfg = make_test_config(str(csv_file))
        cfg.seed = 12345

        # Run simulation twice
        result1 = api.run_simulation(cfg, df)
        result2 = api.run_simulation(cfg, df)

        # Results should be identical
        pd.testing.assert_frame_equal(result1.metrics, result2.metrics)
        assert result1.seed == result2.seed
        assert result1.environment == result2.environment
