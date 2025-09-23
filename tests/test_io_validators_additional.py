from __future__ import annotations

import io

import pandas as pd
import pytest

from trend_analysis.io.validators import (
    ValidationResult,
    create_sample_template,
    detect_frequency,
    load_and_validate_upload,
    validate_returns_schema,
)


def test_validation_result_report_includes_metadata() -> None:
    result = ValidationResult(
        True,
        issues=[],
        warnings=["minor"],
        frequency="monthly",
        date_range=("2024-01-31", "2024-12-31"),
    )

    report = result.get_report()

    assert "✅" in report
    assert "monthly" in report
    assert "2024-01-31" in report
    assert "🟡" in report


def test_validation_result_report_handles_failures() -> None:
    result = ValidationResult(False, issues=["missing"], warnings=["note"])

    report = result.get_report()

    assert "❌" in report
    assert "missing" in report
    assert "🟡" in report


def test_validation_result_report_handles_empty_metadata() -> None:
    result = ValidationResult(True, issues=[], warnings=[])

    report = result.get_report()

    assert "✅" in report
    assert "Det" not in report


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("D", "daily"),
        ("W-MON", "weekly"),
        ("Q", "quarterly"),
        ("A", "annual"),
    ],
)
def test_detect_frequency_known(freq: str, expected: str) -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq=freq)
    df = pd.DataFrame(index=idx)

    assert detect_frequency(df) == expected


def test_detect_frequency_unknown_for_short_series() -> None:
    df = pd.DataFrame(index=pd.DatetimeIndex([]))

    assert detect_frequency(df) == "unknown"


def test_detect_frequency_unknown_for_constant_timestamps() -> None:
    idx = pd.DatetimeIndex([pd.NaT, pd.NaT])
    df = pd.DataFrame(index=idx)

    assert detect_frequency(df) == "unknown"


def test_detect_frequency_irregular_returns_days() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-10", "2020-03-15"])
    df = pd.DataFrame(index=idx)

    out = detect_frequency(df)
    assert out.startswith("irregular")


def test_validate_returns_schema_handles_nat_values() -> None:
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.NaT],
            "Fund": [0.1, 0.2],
        }
    )

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert any("invalid dates" in issue for issue in result.issues)


def test_validate_returns_schema_reports_invalid_strings() -> None:
    df = pd.DataFrame({"Date": ["2020-01-31"] + ["bad"] * 6, "Fund": [0.1] + [0.2] * 6})

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert any("invalid dates" in issue for issue in result.issues)


def test_validate_returns_schema_handles_numeric_warnings_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-03-31"],
            "FundA": [1.0, 1.1, 1.2, 1.3],
            "FundB": [0.1, None, None, None],
        }
    )

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert any(">50%" in warning for warning in result.warnings)


def test_validate_returns_schema_detects_duplicate_dates() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31"] * 7,
            "FundA": list(range(7)),
        }
    )

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert any("Duplicate dates" in issue for issue in result.issues)


def test_validate_returns_schema_numeric_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pd.to_numeric

    def flaky_to_numeric(values, *args, **kwargs):  # pragma: no cover - helper
        if getattr(values, "name", "") == "FundA":
            raise RuntimeError("bad column")
        return original(values, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", flaky_to_numeric)

    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": ["x", "y"],
            "FundB": [1.0, 2.0],
        }
    )

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert any("cannot be converted" in issue for issue in result.issues)


def test_validate_returns_schema_generates_metadata_warnings() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "FundA": [0.1, None, None],
            "FundB": [0.2, 0.3, 0.4],
        }
    )

    result = validate_returns_schema(df)

    assert result.is_valid is True
    assert any(">50%" in warning for warning in result.warnings)


def test_validate_returns_schema_requires_numeric_columns() -> None:
    df = pd.DataFrame({"Date": ["2020-01-31", "2020-02-29"]})

    result = validate_returns_schema(df)

    assert result.is_valid is False
    assert "No numeric" in result.issues[0]


def test_load_and_validate_upload_reads_csv(tmp_path: pd.Series) -> None:
    csv = tmp_path / "sample.csv"
    csv.write_text("Date,Fund\n2020-01-31,0.1\n2020-02-29,0.2\n")

    df, meta = load_and_validate_upload(csv)

    assert "Fund" in df.columns
    assert meta["validation"].is_valid is True


def test_load_and_validate_upload_reads_excel_like_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_read_excel(stream):
        captured["called"] = True
        return pd.DataFrame({"Date": ["2020-01-31"], "Fund": [0.1]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    file_like = io.BytesIO(b"fake")
    file_like.name = "data.xlsx"

    df, _ = load_and_validate_upload(file_like)

    assert captured["called"] is True
    assert "Fund" in df.columns


def test_load_and_validate_upload_handles_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_csv(_stream):
        raise pd.errors.ParserError("broken")

    monkeypatch.setattr(pd, "read_csv", broken_csv)

    buf = io.StringIO("Date,Fund\n2020-01-31,0.1")
    buf.name = "bad.csv"

    with pytest.raises(ValueError, match="Failed to parse"):
        load_and_validate_upload(buf)


def test_load_and_validate_upload_reads_excel_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pd.Series
) -> None:
    captured = {}

    def fake_read_excel(path):
        captured["path"] = path
        return pd.DataFrame({"Date": ["2020-01-31"], "Fund": [0.1]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    excel_path = tmp_path / "sheet.xlsx"
    excel_path.write_bytes(b"fake")

    df, _ = load_and_validate_upload(excel_path)

    assert captured["path"] == excel_path
    assert "Fund" in df.columns


def test_validate_returns_schema_exception_block_guard() -> None:
    # DataFrame with malformed dates
    df = pd.DataFrame({"Date": ["bad"] * 6, "Fund": [1.0] * 6, "Benchmark": [1.0] * 6})
    result = validate_returns_schema(df)
    # Should report issues about invalid dates
    assert any(
        "invalid dates" in issue or "malformed" in issue for issue in result.issues
    )


def test_validate_returns_schema_exception_block_guard_no_issues() -> None:
    # DataFrame with valid dates
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "Fund": [1.0, 2.0],
            "Benchmark": [1.0, 2.0],
        }
    )
    issues = validate_returns_schema(df)
    # Should not report any issues
    assert result.is_valid is True
    assert result.issues == []



def test_create_sample_template_has_expected_shape() -> None:
    template = create_sample_template()

    assert "Date" in template.columns
    assert template.shape[1] >= 3
