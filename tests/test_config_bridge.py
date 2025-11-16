from __future__ import annotations

from pathlib import Path

from trend_analysis.config.bridge import build_config_payload, validate_payload


def test_build_config_payload_minimal():
    payload = build_config_payload(
        csv_path="/tmp/data.csv",
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=5.0,
        target_vol=0.1,
    )
    assert payload["data"]["date_column"] == "Date"
    assert payload["portfolio"]["rebalance_calendar"] == "NYSE"


def test_build_config_payload_optional_entries() -> None:
    payload = build_config_payload(
        csv_path=None,
        universe_membership_path=None,
        managers_glob="data/*.csv",
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=5.0,
        target_vol=0.1,
    )
    assert "csv_path" not in payload["data"]
    assert payload["data"]["managers_glob"] == "data/*.csv"


def test_validate_payload_success(tmp_path: Path):
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path=str(csv),
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=10.0,
        target_vol=0.1,
    )
    validated, error = validate_payload(payload, base_path=tmp_path)
    assert error is None
    assert validated is not None
    assert validated["data"]["csv_path"].endswith("returns.csv")


def test_validate_payload_normalises_path_objects(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path=str(csv),
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=10.0,
        target_vol=0.1,
    )
    payload["data"]["csv_path"] = csv  # emulate caller providing Path object
    validated, error = validate_payload(payload, base_path=tmp_path)
    assert error is None
    assert isinstance(validated["data"]["csv_path"], str)


def test_validate_payload_reports_error(tmp_path: Path):
    payload = build_config_payload(
        csv_path=str(tmp_path / "missing.csv"),
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=10.0,
        target_vol=0.1,
    )
    validated, error = validate_payload(payload, base_path=tmp_path)
    assert validated is None
    assert error is not None
    assert "does not exist" in error or "missing" in error.lower()


def test_validate_payload_invalid_frequency(tmp_path: Path):
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path=str(csv),
        universe_membership_path=None,
        managers_glob=None,
        date_column="Date",
        frequency="Quarterly",  # invalid
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=10.0,
        target_vol=0.1,
    )
    validated, error = validate_payload(payload, base_path=tmp_path)
    assert validated is None
    assert error is not None
    assert "frequency" in error.lower()
def test_build_config_payload_includes_membership_path(tmp_path: Path) -> None:
    membership = tmp_path / "membership.csv"
    membership.write_text("fund,effective_date\nA,2020-01-31\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path="/tmp/data.csv",
        universe_membership_path=str(membership),
        managers_glob=None,
        date_column="Date",
        frequency="M",
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=5.0,
        target_vol=0.1,
    )
    assert payload["data"]["universe_membership_path"] == str(membership)
