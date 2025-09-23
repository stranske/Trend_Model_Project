from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest

from pathlib import Path

from trend_analysis.io import validators


def test_load_and_validate_upload_missing_file() -> None:
    missing = Path("missing.csv")

    with pytest.raises(ValueError, match="File not found: 'missing.csv'"):
        validators.load_and_validate_upload(missing)


def test_load_and_validate_upload_parser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "broken.csv"

    def boom(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad format")

    monkeypatch.setattr(validators.pd, "read_csv", boom)

    with pytest.raises(ValueError, match="Failed to parse file"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_schema_failure() -> None:
    buffer = io.StringIO("Foo,Bar\n1,2\n")
    buffer.name = "invalid.csv"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Schema validation failed"):
        validators.load_and_validate_upload(buffer)


def test_load_and_validate_upload_empty_file(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "empty.csv"

    def empty_reader(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("no data")

    monkeypatch.setattr(validators.pd, "read_csv", empty_reader)

    with pytest.raises(ValueError, match="File contains no data"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "restricted.csv"

    def permission_denied(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise PermissionError("denied")

    monkeypatch.setattr(validators.pd, "read_csv", permission_denied)

    with pytest.raises(ValueError, match="Permission denied accessing file"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_directory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "data.csv"

    def directory_error(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise IsADirectoryError("is directory")

    monkeypatch.setattr(validators.pd, "read_csv", directory_error)

    with pytest.raises(ValueError, match="Path is a directory"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_generic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFile:
        name = "weird.csv"

    def boom(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(validators.pd, "read_csv", boom)

    with pytest.raises(ValueError, match="Failed to read file: 'weird.csv'"):
        validators.load_and_validate_upload(DummyFile())


def test_load_and_validate_upload_excel_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyExcel:
        name = "broken.xlsx"

        def __init__(self) -> None:
            self._buffer = b"excel"

        def read(self) -> bytes:
            return self._buffer

        def seek(self, _pos: int) -> None:
            pass

    def excel_failure(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("no excel data")

    monkeypatch.setattr(validators.pd, "read_excel", excel_failure)

    with pytest.raises(ValueError, match="File contains no data: 'broken.xlsx'"):
        validators.load_and_validate_upload(DummyExcel())


def test_validate_returns_schema_reports_missing_date_column() -> None:
    frame = pd.DataFrame({"FundA": [0.01, 0.02]})
    result = validators.validate_returns_schema(frame)
    assert result.is_valid is False
    assert any("Missing required 'Date' column" in issue for issue in result.issues)


def test_validate_returns_schema_detects_duplicate_dates() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-01-31", "2020-02-29"],
            "FundA": [0.01, 0.02, 0.03],
        }
    )
    result = validators.validate_returns_schema(frame)
    assert result.is_valid is False
    assert any("Duplicate dates" in issue for issue in result.issues)


def test_validate_returns_schema_flags_non_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": ["bad", "data"],
            "FundB": [0.1, 0.2],
        }
    )
    result = validators.validate_returns_schema(frame)
    assert result.is_valid is False
    assert any("Column 'FundA'" in issue for issue in result.issues)


def test_validate_returns_schema_emits_small_sample_warning() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "FundA": [0.01, 0.02],
            "FundB": [0.03, 0.04],
        }
    )
    result = validators.validate_returns_schema(frame)
    assert result.is_valid is True
    assert any("Dataset is quite small" in warning for warning in result.warnings)
