import io
from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_apply_original_headers_round_trips_duplicates() -> None:
    df = pd.DataFrame([[1, 2]], columns=["X", "Y"])

    applied = data_schema.apply_original_headers(df, ["Dup", "Dup"])

    assert applied == ["Dup", "Dup"]
    assert list(df.columns) == ["Dup", "Dup"]


def test_sanitize_formula_headers_renames_and_records_changes() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["=SUM(A1)", "+profit", "Date"])

    sanitized, changes = data_schema._sanitize_formula_headers(df)

    assert list(sanitized.columns) == ["SUM(A1)", "profit", "Date"]
    assert changes == [
        {"original": "=SUM(A1)", "sanitized": "SUM(A1)"},
        {"original": "+profit", "sanitized": "profit"},
    ]


def test_sanitize_formula_headers_noop_preserves_identity() -> None:
    df = pd.DataFrame([[1, 2]], columns=["Clean", "Values"])

    sanitized, changes = data_schema._sanitize_formula_headers(df)

    assert sanitized is df
    assert changes == []


def test_needs_formula_sanitization_handles_whitespace_prefixes() -> None:
    assert data_schema._needs_formula_sanitization("   -growth") is True
    assert data_schema._needs_formula_sanitization("profit") is False


def test_read_binary_payload_round_trips_file_like() -> None:
    payload = io.BytesIO(b"abc")
    payload.name = "upload.bin"
    payload.seek(1)

    raw, name = data_schema._read_binary_payload(payload)

    assert raw == b"abc"
    assert name == "upload.bin"
    assert payload.tell() == 1


def test_read_binary_payload_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError):
        data_schema._read_binary_payload(object())


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


def test_extract_headers_from_excel_failure_returns_none() -> None:
    headers = data_schema.extract_headers_from_bytes(b"garbled", is_excel=True)

    assert headers is None


def test_build_validation_report_populates_all_warnings():
    frame = pd.DataFrame({"A": [1, None, None, None, None], "B": [1, 2, 3, 4, 5]})
    meta = SimpleNamespace(
        columns=["A", "B"],
        symbols=["A", "B"],
        rows=5,
        mode=SimpleNamespace(value="returns"),
        frequency_label="Monthly",
        frequency="M",
        frequency_detected=True,
        frequency_missing_periods=2,
        frequency_max_gap_periods=1,
        frequency_tolerance_periods=0,
        missing_policy="drop",
        missing_policy_limit=0.5,
        missing_policy_summary="dropped A",
        missing_policy_filled=["A"],
        missing_policy_dropped=["A"],
        date_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-05-31")),
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-05-31"),
    )
    validated = SimpleNamespace(frame=frame, metadata=meta)

    report = data_schema._build_validation_report(
        validated, sanitized_columns=[{"original": "=A", "sanitized": "A"}]
    )

    warnings = "\n".join(report["warnings"])
    assert "quite small" in warnings
    assert "50% missing" in warnings
    assert "missing-data" in warnings.lower()
    assert "Sanitized column headers" in warnings


def test_validate_df_wraps_input_validation_error(monkeypatch):
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1]})

    def boom(*args, **kwargs):
        raise data_schema.InputValidationError("boom", issues=["bad"])

    monkeypatch.setattr(data_schema, "validate_input", boom)
    monkeypatch.setattr(data_schema, "validate_market_data", lambda frame: None)

    with pytest.raises(data_schema.MarketDataValidationError) as excinfo:
        data_schema._validate_df(df)

    assert "boom" in str(excinfo.value)
    assert excinfo.value.issues == ["bad"]


def test_load_and_validate_file_uses_excel_branch(monkeypatch):
    dummy_frame = pd.DataFrame({"Date": ["2020-01-01"], "Equity": [0.1]})
    monkeypatch.setattr(data_schema, "validate_input", lambda frame, schema: frame)

    class DummyValidated:
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame
            self.metadata = SimpleNamespace(
                columns=list(frame.columns),
                symbols=list(frame.columns[1:]),
                rows=len(frame),
                mode=SimpleNamespace(value="returns"),
                frequency_label="Monthly",
                frequency="M",
                frequency_detected=True,
                frequency_missing_periods=0,
                frequency_max_gap_periods=0,
                frequency_tolerance_periods=0,
                missing_policy="none",
                missing_policy_limit=None,
                missing_policy_summary="",
                missing_policy_filled=[],
                missing_policy_dropped=[],
                date_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
                start=pd.Timestamp("2020-01-01"),
                end=pd.Timestamp("2020-01-01"),
            )

    monkeypatch.setattr(
        data_schema, "validate_market_data", lambda frame: DummyValidated(frame)
    )
    read_excel_calls: list[str] = []

    def fake_read_excel(buffer):
        read_excel_calls.append(getattr(buffer, "name", ""))
        return dummy_frame.copy()

    monkeypatch.setattr(data_schema.pd, "read_excel", fake_read_excel)

    payload = io.BytesIO(b"excel-bytes")
    payload.name = "file.xlsx"

    frame, meta = data_schema.load_and_validate_file(payload)

    assert list(frame.columns) == ["Date", "Equity"]
    assert read_excel_calls == ["file.xlsx"]
    assert meta["n_rows"] == 1
