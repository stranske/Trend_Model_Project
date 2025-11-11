"""Additional coverage for trend_analysis.io.validators."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from trend_analysis.io import validators
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    MissingPolicyFillDetails,
)
from trend_analysis.io.validators import _read_uploaded_file, _ValidationSummary


class _PathLikeNoRead(os.PathLike[str]):
    """Path-like wrapper with a ``name`` attribute but no ``read`` method."""

    def __init__(self, path: os.PathLike[str]) -> None:
        self._path = os.fspath(path)
        self.name = os.path.basename(self._path)

    def __fspath__(self) -> str:
        return self._path


@pytest.fixture
def summary_metadata() -> MarketDataMetadata:
    """Metadata fixture exercising all summary warning branches."""

    start = pd.Timestamp("2024-01-31")
    end = pd.Timestamp("2024-04-30")
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_detected="M",
        frequency_label="monthly",
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
        missing_policy="zero",
        missing_policy_dropped=["Sparse"],
        missing_policy_summary="Dropped Sparse using zero policy",
        missing_policy_filled={
            "Core": MissingPolicyFillDetails(method="ffill", count=2)
        },
        start=start,
        end=end,
        rows=4,
        columns=["Core", "Sparse"],
    )


def test_validation_summary_reports_all_conditions(
    summary_metadata: MarketDataMetadata,
) -> None:
    frame = pd.DataFrame(
        {
            "Core": [0.1, 0.2, 0.1, 0.15],
            "Sparse": [None, None, 0.05, None],
        }
    )
    summary = _ValidationSummary(summary_metadata, frame)
    warnings = summary.warnings()
    assert any("quite small" in warning for warning in warnings)
    assert any("50%" in warning for warning in warnings)
    assert any("missing monthly periods" in warning for warning in warnings)
    assert any("Dropped Sparse" in warning for warning in warnings)
    assert any("Missing-data policy applied" in warning for warning in warnings)


def test_validation_result_get_report_handles_metadata(
    summary_metadata: MarketDataMetadata,
) -> None:
    result = validators.ValidationResult(
        True,
        issues=[],
        warnings=["warn"],
        frequency=summary_metadata.frequency_label,
        date_range=summary_metadata.date_range,
        metadata=summary_metadata,
    )
    result.mode = MarketDataMode.RETURNS
    report = result.get_report()
    assert "✅" in report and "📊" in report and "📅" in report and "📈" in report
    assert "Missing data policy" in report


def test_validation_result_get_report_failure_lists_issues() -> None:
    result = validators.ValidationResult(False, ["bad"], ["warn"])
    report = result.get_report()
    assert "❌" in report
    assert "🔴" in report and "bad" in report


def test_validation_result_get_report_without_optional_fields() -> None:
    result = validators.ValidationResult(True, [], [])
    report = result.get_report()
    assert report.startswith("✅ Schema validation passed!")
    assert "📊" not in report and "📅" not in report and "📈" not in report


def test_detect_frequency_handles_irregular(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_irregular(index: pd.Index) -> dict:
        raise MarketDataValidationError("Irregular cadence")

    monkeypatch.setattr(validators, "classify_frequency", raise_irregular)
    idx = pd.date_range("2024-01-01", periods=4, freq="2D")
    df = pd.DataFrame(index=idx)
    assert validators.detect_frequency(df).startswith("irregular (")


def test_detect_frequency_handles_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(index: pd.Index) -> dict:
        raise MarketDataValidationError("Some failure")

    monkeypatch.setattr(validators, "classify_frequency", raise_error)
    idx = pd.date_range("2024-01-01", periods=4, freq="2D")
    df = pd.DataFrame(index=idx)
    assert validators.detect_frequency(df) == "unknown"


def test_detect_frequency_uses_code_when_label_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validators,
        "classify_frequency",
        lambda index: {"label": "unknown", "code": "M"},
    )
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(index=idx)
    assert validators.detect_frequency(df) == "M"


def test_detect_frequency_returns_label() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="W")
    df = pd.DataFrame(index=idx)
    assert validators.detect_frequency(df) == "weekly"


def test_read_uploaded_file_reads_excel_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    excel_path = tmp_path / "data.xlsx"
    excel_path.write_bytes(b"")
    fake_frame = pd.DataFrame({"A": [1]})
    monkeypatch.setattr(pd, "read_excel", lambda path: fake_frame)
    frame, source = _read_uploaded_file(str(excel_path))
    assert frame.equals(fake_frame)
    assert source.endswith("data.xlsx")


def test_read_uploaded_file_reads_parquet_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")
    fake_frame = pd.DataFrame({"B": [2]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: fake_frame)
    frame, source = _read_uploaded_file(str(parquet_path))
    assert frame.equals(fake_frame)
    assert source.endswith("data.parquet")


def test_read_uploaded_file_path_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("oops")

    def raise_runtime(path: str) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", raise_runtime)
    with pytest.raises(ValueError, match="Failed to read file"):
        _read_uploaded_file(str(csv_path))


@pytest.mark.parametrize(
    "exc_factory, message",
    [
        (FileNotFoundError, "File not found"),
        (PermissionError, "Permission denied"),
        (IsADirectoryError, "directory"),
        (pd.errors.EmptyDataError, "contains no data"),
        (pd.errors.ParserError, "Failed to parse"),
        (RuntimeError, "Failed to read"),
    ],
)
def test_read_uploaded_file_buffer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory: Callable[..., Exception],
    message: str,
) -> None:
    buffer = io.BytesIO(b"input")
    buffer.name = "broken.csv"

    def raise_exc(*_: object, **__: object) -> pd.DataFrame:
        raise exc_factory("fail")

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError, match=message):
        _read_uploaded_file(buffer)


def test_read_uploaded_file_buffer_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO(b"data")
    buffer.name = "upload.parquet"
    fake_frame = pd.DataFrame({"C": [3]})

    def fake_read(_buf: io.BytesIO) -> pd.DataFrame:
        return fake_frame

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    frame, source = _read_uploaded_file(buffer)
    assert frame.equals(fake_frame)
    assert source == "upload.parquet"


def test_read_uploaded_file_fallback_uses_pathlike(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"x": [1, 2, 3]})
    df.to_csv(csv_path, index=False)
    wrapper = _PathLikeNoRead(csv_path)
    frame, source = _read_uploaded_file(wrapper)
    assert frame.equals(df)
    assert source == wrapper.name


@pytest.mark.parametrize(
    "exc, message",
    [
        (FileNotFoundError("no file"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "directory"),
        (pd.errors.EmptyDataError("empty"), "contains no data"),
        (pd.errors.ParserError("parse"), "Failed to parse"),
        (RuntimeError("boom"), "Failed to read"),
    ],
)
def test_read_uploaded_file_fallback_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    message: str,
) -> None:
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("a,b\n1,2")
    wrapper = _PathLikeNoRead(csv_path)

    def raise_exc(*args, **kwargs):
        raise exc

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError, match=message):
        _read_uploaded_file(wrapper)


def test_read_uploaded_file_rejects_unknown_source() -> None:
    class Mystery:
        pass

    with pytest.raises(ValueError, match="Unsupported upload source"):
        _read_uploaded_file(Mystery())
