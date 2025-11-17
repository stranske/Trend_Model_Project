"""Unit tests for the lightweight CLI/app configuration schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from trend.config_schema import (
    CoreConfigError,
    load_core_config,
    validate_core_config,
)


def _payload(csv_path: Path, membership_path: Path | None = None) -> dict:
    data: dict[str, object] = {
        "csv_path": str(csv_path),
        "date_column": "Date",
        "frequency": "M",
    }
    if membership_path is not None:
        data["universe_membership_path"] = str(membership_path)
    return {
        "data": data,
        "portfolio": {
            "transaction_cost_bps": 5.0,
            "cost_model": {"bps_per_trade": 5.0, "slippage_bps": 0.5},
        },
    }


def test_validate_core_config_round_trips(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    membership = tmp_path / "membership.csv"
    membership.write_text("fund,date\nA,2020-01-31\n", encoding="utf-8")

    result = validate_core_config(
        _payload(csv, membership_path=membership), base_path=tmp_path
    )
    assert result.data.csv_path == csv.resolve()
    assert result.data.universe_membership_path == membership.resolve()
    assert result.data.frequency == "M"
    assert result.costs.transaction_cost_bps == pytest.approx(5.0)

    round_trip = result.to_payload()
    assert round_trip["data"]["csv_path"].endswith("returns.csv")
    assert round_trip["portfolio"]["transaction_cost_bps"] == pytest.approx(5.0)


def test_validate_core_config_requires_csv_path(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "missing.csv")
    payload["data"].pop("csv_path")
    with pytest.raises(CoreConfigError, match=r"data\.csv_path"):
        validate_core_config(payload, base_path=tmp_path)


def test_validate_core_config_rejects_invalid_frequency(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = _payload(csv)
    payload["data"]["frequency"] = "Quarterly"
    with pytest.raises(CoreConfigError, match="frequency"):
        validate_core_config(payload, base_path=tmp_path)


def test_validate_core_config_rejects_cost_type(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = _payload(csv)
    payload["portfolio"]["transaction_cost_bps"] = "not-a-number"
    with pytest.raises(CoreConfigError, match="transaction_cost_bps"):
        validate_core_config(payload, base_path=tmp_path)


def test_load_core_config_reads_yaml(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        """
        data:
          csv_path: returns.csv
          date_column: Date
          frequency: M
        portfolio:
          transaction_cost_bps: 1.5
          cost_model:
            bps_per_trade: 1.5
            slippage_bps: 0.25
        """,
        encoding="utf-8",
    )

    result = load_core_config(cfg_path)
    assert result.data.csv_path == csv.resolve()
    assert result.costs.transaction_cost_bps == pytest.approx(1.5)
