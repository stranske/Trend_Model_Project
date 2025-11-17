import io

import pytest

from streamlit_app.components.csv_validation import (
    CSVValidationError,
    validate_uploaded_csv,
)


def _csv(rows: list[str]) -> bytes:
    header = "Date,Mgr_A,Mgr_B"
    payload = "\n".join([header] + rows)
    return payload.encode("utf-8")


def test_validate_uploaded_csv_accepts_valid_payload() -> None:
    content = _csv(["2020-01-31,0.01,0.02", "2020-02-29,-0.03,0.04"])
    validate_uploaded_csv(content, ("Date",), max_rows=10)


def test_validate_uploaded_csv_enforces_row_cap() -> None:
    rows = [f"2020-01-{idx:02d},0.0,0.1" for idx in range(1, 5)]
    content = _csv(rows)
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(content, ("Date",), max_rows=2)
    assert "limit" in " ".join(excinfo.value.issues)


def test_validate_uploaded_csv_flags_missing_required_column() -> None:
    content = _csv(["2020-01-31,0.01,0.02"])
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(content, ("Date", "Mgr_C"), max_rows=10)
    assert "Missing columns" in excinfo.value.issues[0]


def test_validate_uploaded_csv_detects_invalid_dates() -> None:
    content = _csv(["not-a-date,0.01,0.02"])
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(content, ("Date",), max_rows=10)
    assert "cannot be parsed" in " ".join(excinfo.value.issues)


def test_validate_uploaded_csv_detects_duplicate_dates() -> None:
    content = _csv(["2020-01-31,0.01,0.02", "2020-01-31,0.03,0.04"])
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(content, ("Date",), max_rows=10)
    assert "repeats the date" in excinfo.value.issues[0]


def test_validate_uploaded_csv_accepts_file_like_object() -> None:
    buffer = io.BytesIO(_csv(["2020-01-31,0.01,0.02"]))
    buffer.name = "custom.csv"
    validate_uploaded_csv(buffer, ("Date",), max_rows=10)


def test_validate_uploaded_csv_reports_empty_dataset() -> None:
    content = "Date".encode("utf-8")
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(content, ("Date",), max_rows=10)
    assert "Detected 0 rows" in " ".join(excinfo.value.issues)
    assert excinfo.value.sample_preview is not None


def test_validate_uploaded_csv_reports_missing_date_column() -> None:
    rows = ["2020-01-31,0.01,0.02"]
    payload = "\n".join(["Mgr_A,Mgr_B"] + rows)
    with pytest.raises(CSVValidationError) as excinfo:
        validate_uploaded_csv(payload.encode("utf-8"), ("Date",), max_rows=10)
    assert "Missing columns" in excinfo.value.issues[0]
    assert excinfo.value.sample_preview is not None
