"""Unit tests for the lightweight CLI/app configuration schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from trend.config_schema import (
    CoreConfigError,
    load_core_config,
    validate_core_config,
)


def _payload(
    csv_path: Path | None,
    *,
    membership_path: Path | None = None,
    managers_glob: str | None = None,
) -> dict:
    data: dict[str, object] = {
        "date_column": "Date",
        "frequency": "M",
    }
    if csv_path is not None:
        data["csv_path"] = str(csv_path)
    if managers_glob is not None:
        data["managers_glob"] = managers_glob
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
    assert result.data.managers_glob is None
    assert result.data.universe_membership_path == membership.resolve()
    assert result.data.frequency == "M"
    assert result.costs.transaction_cost_bps == pytest.approx(5.0)

    round_trip = result.to_payload()
    assert round_trip["data"]["csv_path"].endswith("returns.csv")
    assert round_trip["portfolio"]["transaction_cost_bps"] == pytest.approx(5.0)


def test_validate_core_config_requires_csv_path(tmp_path: Path) -> None:
    payload = _payload(None)
    with pytest.raises(CoreConfigError, match="Provide data.csv_path or"):
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


def test_validate_core_config_allows_managers_glob(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    csv = data_dir / "mgr_A.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = _payload(
        None,
        managers_glob=str((tmp_path / "inputs" / "mgr_*.csv").relative_to(tmp_path)),
    )
    result = validate_core_config(payload, base_path=tmp_path)
    assert result.data.csv_path is None
    assert result.data.managers_glob.endswith("mgr_*.csv")


def test_validate_core_config_rejects_missing_managers_glob(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    payload = _payload(
        None,
        managers_glob=str((data_dir / "*.csv").relative_to(tmp_path)),
    )
    with pytest.raises(CoreConfigError, match="did not match any files"):
        validate_core_config(payload, base_path=tmp_path)


def test_load_core_config_reads_yaml(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        """
        data:
          managers_glob: returns.csv
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
    assert result.data.managers_glob.endswith("returns.csv")
    assert result.costs.transaction_cost_bps == pytest.approx(1.5)
