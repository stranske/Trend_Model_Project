"""Additional coverage for ``trend_analysis.io.validators``."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io import validators
from trend_analysis.io.market_data import MarketDataMetadata, MarketDataMode, MarketDataValidationError


def _metadata(**overrides: object) -> MarketDataMetadata:
    base = dict(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_label="Monthly",
        start=datetime(2024, 1, 31),
        end=datetime(2024, 2, 29),
        rows=8,
        columns=["Fund_A", "Fund_B"],
        missing_policy="drop",
    )
    base.update(overrides)
    return MarketDataMetadata(**base)


def test_validation_summary_warns_for_small_dataset() -> None:
    frame = pd.DataFrame({"Fund_A": [1.0, None, None], "Fund_B": [None, None, None]})
    metadata = _metadata(
        rows=3,
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
        missing_policy_dropped=["Fund_B"],
        missing_policy_summary="Filled gaps",
    )
    summary = validators._ValidationSummary(metadata, frame)
    warnings = summary.warnings()
    assert any("Dataset is quite small" in w for w in warnings)
    assert any("Column 'Fund_B'" in w for w in warnings)
    assert any("missing" in w.lower() and "2" in w for w in warnings)
    assert any("Missing-data policy dropped" in w for w in warnings)
    assert any("Missing-data policy applied" in w for w in warnings)


def test_validation_result_report_includes_metadata() -> None:
    metadata = _metadata(
        rows=12,
        start=datetime(2024, 1, 31),
        end=datetime(2024, 2, 29),
        frequency_missing_periods=1,
        missing_policy_summary="Applied",
    )
    result = validators.ValidationResult(
        True,
        issues=[],
        warnings=["Warn"],
        frequency="Monthly",
        date_range=metadata.date_range,
        metadata=metadata,
    )
    report = result.get_report()
    assert "Detected frequency" in report
    assert "Date range" in report
    assert "Missing data policy" in report
    assert "Warn" in report


def test_detect_frequency_handles_irregular(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_irregular(_index: pd.Index) -> dict[str, object]:
        raise MarketDataValidationError("Irregular cadence", issues=[])

    monkeypatch.setattr(validators, "classify_frequency", raise_irregular)
    series = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-31", periods=2, freq="M"))
    label = validators.detect_frequency(series.to_frame())
    assert "irregular" in label.lower()


def test_detect_frequency_unknown_when_exception_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_generic(_index: pd.Index) -> dict[str, object]:
        raise MarketDataValidationError("bad", issues=[])

    monkeypatch.setattr(validators, "classify_frequency", raise_generic)
    series = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-31", periods=2, freq="M"))
    assert validators.detect_frequency(series.to_frame()) == "unknown"


def test_detect_frequency_uses_code_when_label_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def return_info(_index: pd.Index) -> dict[str, object]:
        return {"label": "unknown", "code": "W"}

    monkeypatch.setattr(validators, "classify_frequency", return_info)
    series = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-31", periods=2, freq="M"))
    assert validators.detect_frequency(series.to_frame()) == "W"


def test_read_uploaded_file_path_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validators._read_uploaded_file(tmp_path / "missing.csv")

    directory = tmp_path / "folder"
    directory.mkdir()
    with pytest.raises(ValueError):
        validators._read_uploaded_file(directory)

    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("1,2,3", encoding="utf-8")
    with pytest.raises(ValueError):
        validators._read_uploaded_file(bad_file.with_suffix(".xlsx"))


@pytest.mark.parametrize(
    "exception, message",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "Path is a directory"),
        (pd.errors.EmptyDataError("empty"), "File contains no data"),
        (pd.errors.ParserError("parse"), "Failed to parse"),
    ],
)
def test_read_uploaded_file_stream_exceptions(
    monkeypatch: pytest.MonkeyPatch, exception: Exception, message: str
) -> None:
    class Dummy(io.StringIO):
        def __init__(self, exc: Exception):
            super().__init__("content")
            self._exc = exc
            self.name = "upload.csv"

    dummy = Dummy(exception)

    def raise_exc(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise dummy._exc

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError) as captured:
        validators._read_uploaded_file(dummy)
    assert message in str(captured.value)


def test_read_uploaded_file_generic_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO("content")
    buffer.name = "upload.csv"

    def boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", boom)
    with pytest.raises(ValueError) as excinfo:
        validators._read_uploaded_file(buffer)
    assert "Failed to read file" in str(excinfo.value)


@pytest.mark.parametrize(
    "exception, message",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "Path is a directory"),
        (pd.errors.EmptyDataError("empty"), "File contains no data"),
        (pd.errors.ParserError("parse"), "Failed to parse"),
        (RuntimeError("boom"), "Failed to read file"),
    ],
)
def test_read_uploaded_file_lower_name_errors(
    monkeypatch: pytest.MonkeyPatch, exception: Exception, message: str
) -> None:
    class NoRead:
        def __init__(self, name: str) -> None:
            self.name = name

    dummy = NoRead("named.csv")

    def raise_exc(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise exception

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError) as excinfo:
        validators._read_uploaded_file(dummy)
    assert message in str(excinfo.value)
