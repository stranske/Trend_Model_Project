from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config.validation import validate_config


def _write_returns(path: Path) -> Path:
    path.write_text("Date,FundA\n2020-01-31,0.01\n2020-02-29,0.02\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, frequency: str) -> dict[str, object]:
    csv_path = _write_returns(tmp_path / "returns.csv")
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": frequency,
            "missing_policy": "drop",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.15},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "transaction_cost_bps": 10,
            "cost_model": {
                "bps_per_trade": 0,
                "slippage_bps": 0,
            },
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


@pytest.mark.parametrize("frequency", ["D", "W"])
def test_non_monthly_data_frequency_is_rejected(tmp_path: Path, frequency: str) -> None:
    result = validate_config(_config(tmp_path, frequency), base_path=tmp_path)

    frequency_errors = [error for error in result.errors if error.path == "data.frequency"]
    assert frequency_errors
    assert any("Only monthly data.frequency" in error.message for error in frequency_errors)
    assert not result.valid


def test_non_monthly_data_frequency_is_rejected_during_cli_preflight() -> None:
    result = validate_config(
        {"version": "1", "data": {"frequency": "D"}},
        skip_required_fields=True,
    )

    assert any(
        error.path == "data.frequency" and "Only monthly data.frequency" in error.message
        for error in result.errors
    )


@pytest.mark.parametrize("frequency", ["M", "ME"])
def test_monthly_data_frequency_remains_valid(tmp_path: Path, frequency: str) -> None:
    result = validate_config(_config(tmp_path, frequency), base_path=tmp_path)

    assert not [error for error in result.errors if error.path == "data.frequency"]
