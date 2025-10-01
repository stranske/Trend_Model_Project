from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io.validators import (
    FREQUENCY_MAP,
    ValidationResult,
    load_and_validate_upload,
    validate_returns_schema,
)


class TestValidationResult:
    def test_valid_report(self) -> None:
        result = ValidationResult(True, [], [], "monthly", ("2024-01-31", "2024-12-31"))
        report = result.get_report()
        assert "✅ Schema validation passed!" in report
        assert "monthly" in report
        assert "2024-01-31" in report and "2024-12-31" in report

    def test_invalid_report(self) -> None:
        result = ValidationResult(False, ["Missing Date"], ["Warn"])
        report = result.get_report()
        assert "❌ Schema validation failed!" in report
        assert "Missing Date" in report
        assert "Warn" in report


class TestValidateReturnsSchema:
    def test_valid_schema(self) -> None:
        df = pd.DataFrame(
            {
                "Date": ["2024-01-31", "2024-02-29"],
                "Fund": [0.01, 0.02],
            }
        )
        result = validate_returns_schema(df)
        assert result.is_valid
        assert result.frequency == "monthly"

    def test_missing_date_column(self) -> None:
        df = pd.DataFrame({"Fund": [0.01, 0.02]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert "Expected a 'Date' column" in result.issues[0]

    def test_duplicate_dates_issue(self) -> None:
        df = pd.DataFrame({"Date": ["2024-01-31", "2024-01-31"], "Fund": [0.01, 0.02]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert "Duplicate" in result.issues[0]

    def test_non_numeric_column(self) -> None:
        df = pd.DataFrame({"Date": ["2024-01-31", "2024-02-29"], "Fund": ["a", "b"]})
        result = validate_returns_schema(df)
        assert not result.is_valid
        assert "Failed to coerce numeric data" in result.issues[0]


class TestLoadAndValidateUpload:
    def test_load_csv_file(self) -> None:
        csv_data = "Date,Fund\n2024-01-31,0.01\n2024-02-29,0.02"
        file_like = io.StringIO(csv_data)
        file_like.name = "test.csv"

        frame, meta = load_and_validate_upload(file_like)
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert meta["frequency"] == "monthly"
        assert meta["validation"].is_valid
        assert meta["mode"] == "returns"

    def test_load_invalid_file(self) -> None:
        csv_data = "Fund\n0.01\n0.02"
        file_like = io.StringIO(csv_data)
        file_like.name = "test.csv"

        with pytest.raises(ValueError) as exc:
            load_and_validate_upload(file_like)
        assert "Schema validation failed" in str(exc.value)

    def test_nonexistent_path(self) -> None:
        with pytest.raises(ValueError) as exc:
            load_and_validate_upload("missing.csv")
        assert "File not found" in str(exc.value)

    def test_directory_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as exc:
            load_and_validate_upload(tmp_path)
        assert "Path is a directory" in str(exc.value)

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as tmp:
            tmp.flush()
            with pytest.raises(ValueError) as exc:
                load_and_validate_upload(tmp.name)
            assert "File contains no data" in str(exc.value)

    def test_parser_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_parser(*args: object, **kwargs: object) -> pd.DataFrame:
            raise pd.errors.ParserError("bad parse")

        monkeypatch.setattr(pd, "read_csv", raise_parser)
        with pytest.raises(ValueError) as exc:
            load_and_validate_upload("bad.csv")
        assert "Failed to parse file" in str(exc.value)

    def test_generic_reader_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("boom")

        monkeypatch.setattr(pd, "read_csv", boom)
        with pytest.raises(ValueError) as exc:
            load_and_validate_upload("broken.csv")
        assert "Failed to read file" in str(exc.value)


class TestFrequencyMap:
    def test_frequency_entries(self) -> None:
        assert FREQUENCY_MAP["monthly"] == "M"
        assert FREQUENCY_MAP["weekly"] == "W"
        assert "irregular" in FREQUENCY_MAP
