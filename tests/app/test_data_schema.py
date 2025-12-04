import io

import pandas as pd

from trend_portfolio_app import data_schema
from trend_portfolio_app.data_schema import load_and_validate_csv


def test_extract_headers_from_csv_bytes_with_bom() -> None:
    raw = "\ufeffDate,A,B\n2020-01-01,1,2\n".encode("utf-8")

    headers = data_schema.extract_headers_from_bytes(raw, is_excel=False)

    assert headers == ["Date", "A", "B"]


def test_apply_original_headers_mismatch_returns_none() -> None:
    df = pd.DataFrame([[1, 2]], columns=["X", "Y"])

    applied = data_schema.apply_original_headers(df, ["A", "B", "C"])

    assert applied is None
    assert list(df.columns) == ["X", "Y"]


def test_sanitize_formula_headers_renames_and_records_changes() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["=SUM(A1)", "+profit", "Date"])

    sanitized, changes = data_schema._sanitize_formula_headers(df)

    assert list(sanitized.columns) == ["SUM(A1)", "profit", "Date"]
    assert changes == [
        {"original": "=SUM(A1)", "sanitized": "SUM(A1)"},
        {"original": "+profit", "sanitized": "profit"},
    ]


def test_read_binary_payload_round_trips_file_like() -> None:
    payload = io.BytesIO(b"abc")
    payload.name = "upload.bin"
    payload.seek(1)

    raw, name = data_schema._read_binary_payload(payload)

    assert raw == b"abc"
    assert name == "upload.bin"
    assert payload.tell() == 1


def test_load_and_validate_csv(tmp_path):
    csv = tmp_path / "toy.csv"
    csv.write_text("Date,A,B\n2020-01-31,0.01,0.02\n2020-02-29,0.00,-0.01\n")
    df, meta = load_and_validate_csv(csv)
    assert set(df.columns) == {"A", "B"}
    assert len(df) == 2


def test_load_and_validate_file_sanitizes_headers_and_builds_meta(
    monkeypatch, tmp_path
):
    csv_path = tmp_path / "formulas.csv"
    csv_path.write_text("=evil,@bad\n1,2\n3,4\n")

    def fake_validate_input(frame, schema):
        return frame

    class DummyMode:
        value = "returns"

    class DummyValidated:
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame
            self.metadata = type(
                "Meta",
                (),
                {
                    "columns": list(frame.columns),
                    "symbols": list(frame.columns),
                    "rows": len(frame),
                    "mode": DummyMode(),
                    "frequency_label": "Monthly",
                    "frequency": "M",
                    "frequency_detected": True,
                    "frequency_missing_periods": 1,
                    "frequency_max_gap_periods": 1,
                    "frequency_tolerance_periods": 0,
                    "missing_policy": "drop",
                    "missing_policy_limit": 0.5,
                    "missing_policy_summary": "dropped bad",
                    "missing_policy_filled": [],
                    "missing_policy_dropped": ["evil"],
                    "date_range": (
                        pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-31"),
                    ),
                    "start": pd.Timestamp("2020-01-01"),
                    "end": pd.Timestamp("2020-01-31"),
                },
            )()

    def fake_validate_market_data(frame):
        return DummyValidated(frame)

    monkeypatch.setattr(data_schema, "validate_input", fake_validate_input)
    monkeypatch.setattr(data_schema, "validate_market_data", fake_validate_market_data)

    frame, meta = data_schema.load_and_validate_file(csv_path)

    assert list(frame.columns) == ["evil", "bad"]
    sanitized_changes = meta.get("sanitized_columns")
    assert {change["original"] for change in sanitized_changes} == {"=evil", "@bad"}
    warnings = meta.get("validation", {}).get("warnings", [])
    assert any("Missing-data policy" in warning for warning in warnings)
    assert any("Sanitized column headers" in warning for warning in warnings)
