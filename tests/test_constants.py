"""Tests for constants module and their usage."""

from pathlib import Path

import pandas as pd

from trend import cli
from trend import cli_owned_commands as owned
from trend_analysis.constants import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_OUTPUT_FORMATS,
    NUMERICAL_TOLERANCE_HIGH,
    NUMERICAL_TOLERANCE_LOW,
    NUMERICAL_TOLERANCE_MEDIUM,
)


def test_constants_exist():
    """Test that constants are properly defined."""
    assert DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert DEFAULT_OUTPUT_FORMATS == ["excel"]

    # Test numerical tolerance constants
    assert NUMERICAL_TOLERANCE_HIGH == 1e-12
    assert NUMERICAL_TOLERANCE_MEDIUM == 1e-9
    assert NUMERICAL_TOLERANCE_LOW == 1e-6


def test_numerical_tolerance_constants_hierarchy():
    """Test that tolerance constants maintain the expected hierarchy."""
    assert NUMERICAL_TOLERANCE_HIGH < NUMERICAL_TOLERANCE_MEDIUM
    assert NUMERICAL_TOLERANCE_MEDIUM < NUMERICAL_TOLERANCE_LOW


def test_constants_are_used_in_cli():
    """Test that cli-owned command helpers use the constants."""
    assert hasattr(owned, "DEFAULT_OUTPUT_DIRECTORY")
    assert hasattr(owned, "DEFAULT_OUTPUT_FORMATS")
    assert owned.DEFAULT_OUTPUT_DIRECTORY == DEFAULT_OUTPUT_DIRECTORY
    assert owned.DEFAULT_OUTPUT_FORMATS == DEFAULT_OUTPUT_FORMATS


def _write_cfg(path: Path, csv: Path) -> None:
    """Helper to write a minimal config file."""
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                f"data: {{csv_path: '{csv}', date_column: 'Date', frequency: 'M', risk_free_column: 'RF', allow_risk_free_fallback: false, missing_policy: drop}}",
                "preprocessing: {}",
                "vol_adjust: {target_vol: 1.0}",
                "sample_split: {in_start: '2020-01', in_end: '2020-03', "
                "out_start: '2020-04', out_end: '2020-06'}",
                "portfolio: {selection_mode: all, rebalance_calendar: NYSE, max_turnover: 0.5, cost_model: {per_trade_bps: 10, half_spread_bps: 0}}",
                "metrics: {}",
                "export: {}",  # Empty export config to trigger defaults
                "run: {}",
            ]
        )
    )


def _make_df():
    """Helper to create a minimal test dataframe."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_constants_default_behavior_unchanged(tmp_path, monkeypatch):
    """Test that the default behavior using constants is unchanged."""
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, csv)
    monkeypatch.chdir(tmp_path)

    # Run the public CLI with empty export config (should use constants for defaults).
    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    assert rc == 0

    # Check that the default output directory and file are created
    assert (tmp_path / DEFAULT_OUTPUT_DIRECTORY / "analysis.xlsx").exists()


def test_numerical_tolerance_constants_in_use():
    """Test that numerical tolerance constants are being imported in key
    modules."""
    # Test that the constants can be imported from specific modules that use them
    from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH as const_tol
    from trend_analysis.engine.optimizer import NUMERICAL_TOLERANCE_HIGH as opt_tol

    # Verify they are the same constant
    assert opt_tol is const_tol
    assert opt_tol == 1e-12

    # Test a few other modules
    from trend_analysis.multi_period.engine import (
        NUMERICAL_TOLERANCE_HIGH as engine_tol,
    )

    assert engine_tol is const_tol
