from __future__ import annotations

from pathlib import Path


from trend_analysis.config.bridge import build_config_payload, validate_payload


def test_build_config_payload_minimal():
    payload = build_config_payload(
        csv_path="/tmp/data.csv",
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


def test_validate_payload_success(tmp_path: Path):
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    payload = build_config_payload(
        csv_path=str(csv),
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


def test_validate_payload_reports_error(tmp_path: Path):
    payload = build_config_payload(
        csv_path=str(tmp_path / "missing.csv"),
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
