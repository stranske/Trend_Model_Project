from pathlib import Path

import pandas as pd
import yaml  # type: ignore[import-untyped]

from trend_analysis import api, cli
from trend_analysis.config import Config


def make_test_data() -> pd.DataFrame:
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

    base_dir = Path(csv_path).parent
    return Config(
        version="1",
        data={"csv_path": csv_path},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        portfolio={},
        benchmarks={"spx": "SPX"},
        metrics={},
        export={"directory": str(base_dir / "noop"), "formats": []},
        run={},
    )


def _write_config(cfg_path: Path, config: Config) -> None:
    """Write Config object to YAML file."""

    config_dict = config.model_dump()
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def test_cli_api_golden_master(tmp_path, capsys):
    """Test that CLI and API produce identical outputs for same inputs."""

    csv_file = tmp_path / "test_data.csv"
    config_file = tmp_path / "test_config.yml"

    df = make_test_data()
    df.to_csv(csv_file, index=False)

    cfg = make_test_config(str(csv_file))
    _write_config(config_file, cfg)

    api_result = api.run_simulation(cfg, df)

    assert isinstance(api_result.metrics, pd.DataFrame)
    assert not api_result.metrics.empty
    assert api_result.summary_text is not None

    rc = cli.main(["run", "-c", str(config_file), "-i", str(csv_file)])
    captured = capsys.readouterr()

    assert rc == 0
    summary_cli = captured.out.strip()
    summary_api = (api_result.summary_text or "").strip()
    assert summary_cli == summary_api

    assert "out_sample_stats" in api_result.details
    assert "benchmark_ir" in api_result.details
    assert "python" in api_result.environment
    assert "numpy" in api_result.environment
    assert "pandas" in api_result.environment


def test_api_deterministic_behavior(tmp_path):
    """Test that API produces deterministic results."""

    csv_file = tmp_path / "test_data.csv"

    df = make_test_data()
    df.to_csv(csv_file, index=False)

    cfg = make_test_config(str(csv_file))
    cfg.seed = 12345

    result1 = api.run_simulation(cfg, df)
    result2 = api.run_simulation(cfg, df)

    pd.testing.assert_frame_equal(result1.metrics, result2.metrics)
    assert result1.seed == result2.seed
    assert result1.environment == result2.environment
