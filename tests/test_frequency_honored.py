from __future__ import annotations

from pathlib import Path
from typing import Any

from trend_analysis.config.validation import validate_config


def _base_config(tmp_path: Path, frequency: str) -> dict[str, Any]:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Date,A,B\n2020-01-31,0.0,0.0\n2020-02-29,0.01,0.02\n",
        encoding="utf-8",
    )
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": frequency,
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0.0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


def _frequency_errors(config: dict[str, Any]) -> list[str]:
    result = validate_config(config, skip_required_fields=True)
    return [issue.message for issue in result.errors if issue.path == "data.frequency"]


def test_daily_frequency_is_rejected_instead_of_silently_monthly(tmp_path: Path) -> None:
    messages = _frequency_errors(_base_config(tmp_path, "D"))

    assert messages
    assert any("silently resampled to monthly" in message for message in messages)


def test_weekly_frequency_is_rejected_instead_of_silently_monthly(tmp_path: Path) -> None:
    messages = _frequency_errors(_base_config(tmp_path, "W"))

    assert messages
    assert any("Only monthly data.frequency values" in message for message in messages)


def test_monthly_frequency_values_remain_valid(tmp_path: Path) -> None:
    assert not _frequency_errors(_base_config(tmp_path, "M"))
    assert not _frequency_errors(_base_config(tmp_path, "ME"))
