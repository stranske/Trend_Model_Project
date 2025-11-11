from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis.io import validators
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    MissingPolicyFillDetails,
    ValidatedMarketData,
)


def _make_metadata(**overrides: Any) -> MarketDataMetadata:
    base = dict(
        mode=MarketDataMode.RETURNS,
        frequency="monthly",
        frequency_label="Monthly",
        start=pd.Timestamp("2023-01-31"),
        end=pd.Timestamp("2023-10-31"),
        rows=10,
        columns=["FundA", "FundB"],
        missing_policy="drop",
        missing_policy_summary="Dropped FundB for missing data",
        missing_policy_dropped=["FundB"],
        missing_policy_filled={
            "FundA": MissingPolicyFillDetails(method="ffill", count=3)
        },
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
    )
    base.update(overrides)
    return MarketDataMetadata(**base)


def test_validation_summary_collects_all_warning_categories() -> None:
    frame = pd.DataFrame(
        {
            "FundA": [1.0] + [np.nan] * 9,
            "FundB": [np.nan] * 10,
        }
    )
    summary = validators._ValidationSummary(_make_metadata(), frame)

    warnings = summary.warnings()

    assert any("quite small" in message for message in warnings)
    assert any("Column 'FundA'" in message for message in warnings)
    assert any("missing Monthly periods" in message for message in warnings)
    assert any("Missing-data policy dropped columns" in message for message in warnings)
    assert any("Missing-data policy applied" in message for message in warnings)


def test_validation_result_get_report_includes_metadata_details() -> None:
    metadata = _make_metadata()
    warnings = ["Column 'FundB' has >50% missing values (0/10 valid)."]

    report = validators.ValidationResult(
        True,
        [],
        warnings,
        frequency=metadata.frequency_label,
        date_range=metadata.date_range,
        metadata=metadata,
    ).get_report()

    assert "Schema validation passed" in report
    assert metadata.frequency_label in report
    assert metadata.date_range[0] in report and metadata.date_range[1] in report
    assert metadata.mode.value in report
    assert metadata.missing_policy_summary in report
    assert "Warnings" in report and warnings[0] in report


def test_validation_result_get_report_handles_failures() -> None:
    report = validators.ValidationResult(
        False,
        ["Missing Date column"],
        ["Small sample size"],
    ).get_report()

    assert "Schema validation failed" in report
    assert "Issues that must be fixed" in report
    assert "Warnings" in report


@pytest.mark.parametrize(
    "index_factory, classify_result, expected",
    [
        (lambda: pd.Index([1, 2, 3]), None, "unknown"),
        (
            lambda: pd.date_range("2023-01-01", periods=3, freq="D"),
            {"label": "Weekly"},
            "Weekly",
        ),
        (
            lambda: pd.date_range("2023-01-01", periods=3, freq="D"),
            {"label": "unknown", "code": "W"},
            "W",
        ),
    ],
)
def test_detect_frequency_prefers_label_then_code(
    monkeypatch: pytest.MonkeyPatch,
    index_factory,
    classify_result,
    expected,
) -> None:
    df = pd.DataFrame(index=index_factory(), data={"value": [1, 2, 3]})

    if classify_result is not None:
        monkeypatch.setattr(
            validators,
            "classify_frequency",
            lambda index: classify_result,
        )

    assert validators.detect_frequency(df) == expected


def test_detect_frequency_formats_irregular_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(index=pd.date_range("2023-01-01", periods=3, freq="D"))

    def raise_irregular(_: pd.DatetimeIndex) -> dict[str, Any]:
        raise MarketDataValidationError("Irregular cadence detected")

    monkeypatch.setattr(validators, "classify_frequency", raise_irregular)

    result = validators.detect_frequency(df)
    assert result.startswith("irregular (")
    assert "Irregular cadence" in result


def test_detect_frequency_handles_generic_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(index=pd.date_range("2023-01-01", periods=3, freq="D"))

    def raise_generic(_: pd.DatetimeIndex) -> dict[str, Any]:
        raise MarketDataValidationError("Calculation failed")

    monkeypatch.setattr(validators, "classify_frequency", raise_generic)

    assert validators.detect_frequency(df) == "unknown"


def test_build_result_wraps_validated_market_data() -> None:
    metadata = _make_metadata(rows=20)
    frame = pd.DataFrame({"FundA": np.linspace(0.0, 1.0, metadata.rows)})

    validated = ValidatedMarketData(frame, metadata)
    result = validators._build_result(validated)

    assert result.is_valid
    assert result.metadata is metadata
    assert isinstance(result.warnings, list)


def test_validate_returns_schema_success(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _make_metadata()
    frame = pd.DataFrame({"FundA": [0.1, 0.2]})
    validated = ValidatedMarketData(frame, metadata)

    monkeypatch.setattr(validators, "validate_market_data", lambda df: validated)

    result = validators.validate_returns_schema(frame)
    assert result.is_valid
    assert result.metadata is metadata


def test_validate_returns_schema_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    error = MarketDataValidationError("Missing values", issues=["FundA all NaN"])

    def raise_error(_: pd.DataFrame) -> ValidatedMarketData:
        raise error

    monkeypatch.setattr(validators, "validate_market_data", raise_error)

    result = validators.validate_returns_schema(pd.DataFrame())
    assert not result.is_valid
    assert result.issues == ["FundA all NaN"]


def test_read_uploaded_file_from_existing_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

    frame, source = validators._read_uploaded_file(csv_path)

    assert source == str(csv_path)
    assert frame["value"].tolist() == [1, 2, 3]


def test_read_uploaded_file_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="File not found"):
        validators._read_uploaded_file(tmp_path / "missing.csv")


def test_read_uploaded_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Path is a directory"):
        validators._read_uploaded_file(tmp_path)


def test_read_uploaded_file_handles_excel_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeUpload(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"binary-data")
            self.name = "report.xlsx"

    fake = FakeUpload()
    expected = pd.DataFrame({"value": [42]})

    def fake_read_excel(buf: io.BytesIO) -> pd.DataFrame:
        assert isinstance(buf, io.BytesIO)
        return expected

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    frame, source = validators._read_uploaded_file(fake)

    assert frame.equals(expected)
    assert source == "report.xlsx"
    assert fake.tell() == 0


def test_read_uploaded_file_handles_parquet_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeParquet(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"parquet-data")
            self.name = "sample.parquet"

    fake = FakeParquet()
    expected = pd.DataFrame({"value": [5, 6]})

    def fake_read_parquet(buf: io.BytesIO) -> pd.DataFrame:
        assert isinstance(buf, io.BytesIO)
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    frame, source = validators._read_uploaded_file(fake)

    assert frame.equals(expected)
    assert source == "sample.parquet"


def test_read_uploaded_file_handles_csv_stream() -> None:
    stream = io.StringIO("value\n1\n")
    stream.name = "upload.csv"

    frame, source = validators._read_uploaded_file(stream)

    assert source == "upload.csv"
    assert frame.equals(pd.DataFrame({"value": [1]}))


def test_read_uploaded_file_path_excel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "workbook.xlsx"
    path.write_bytes(b"dummy")
    expected = pd.DataFrame({"value": [11]})

    monkeypatch.setattr(pd, "read_excel", lambda p: expected)

    frame, source = validators._read_uploaded_file(path)
    assert frame.equals(expected)
    assert source == str(path)


def test_read_uploaded_file_path_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"dummy")
    expected = pd.DataFrame({"value": [3, 4]})

    monkeypatch.setattr(pd, "read_parquet", lambda p: expected)

    frame, source = validators._read_uploaded_file(path)
    assert frame.equals(expected)
    assert source == str(path)


def test_read_uploaded_file_path_generic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("value\n1\n")

    def raise_runtime(_: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", raise_runtime)

    with pytest.raises(ValueError, match="Failed to read file"):
        validators._read_uploaded_file(path)


@pytest.mark.parametrize(
    "exception, message",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "Path is a directory"),
        (pd.errors.EmptyDataError("empty"), "contains no data"),
        (pd.errors.ParserError("bad"), "Failed to parse"),
        (RuntimeError("boom"), "Failed to read file"),
    ],
)
def test_read_uploaded_file_translates_stream_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    message: str,
) -> None:
    class Faulty(io.StringIO):
        def __init__(self) -> None:
            super().__init__("")
            self.name = "faulty.csv"

        def read(self, *args: Any, **kwargs: Any) -> str:
            raise exception

    stream = Faulty()

    with pytest.raises(ValueError, match=message):
        validators._read_uploaded_file(stream)


def test_read_uploaded_file_named_without_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NamedOnly:
        def __init__(self) -> None:
            self.name = "fallback.csv"

    expected = pd.DataFrame({"value": [7]})

    def fake_read_csv(obj: Any) -> pd.DataFrame:
        assert isinstance(obj, NamedOnly)
        return expected

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    frame, source = validators._read_uploaded_file(NamedOnly())
    assert frame.equals(expected)
    assert source == "fallback.csv"


def test_read_uploaded_file_fallback_translates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NamedOnly:
        def __init__(self) -> None:
            self.name = "broken.csv"

    def raise_parser(_: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("broken")

    monkeypatch.setattr(pd, "read_csv", raise_parser)

    with pytest.raises(ValueError, match="Failed to parse"):
        validators._read_uploaded_file(NamedOnly())


@pytest.mark.parametrize(
    "exception, message",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "Path is a directory"),
        (pd.errors.EmptyDataError("empty"), "contains no data"),
        (RuntimeError("boom"), "Failed to read file"),
    ],
)
def test_read_uploaded_file_fallback_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    message: str,
) -> None:
    class NamedOnly:
        def __init__(self) -> None:
            self.name = "fallback.csv"

    def raise_error(_: Any) -> pd.DataFrame:
        raise exception

    monkeypatch.setattr(pd, "read_csv", raise_error)

    with pytest.raises(ValueError, match=message):
        validators._read_uploaded_file(NamedOnly())


def test_read_uploaded_file_rejects_unknown_sources() -> None:
    with pytest.raises(ValueError, match="Unsupported upload source"):
        validators._read_uploaded_file(object())


def test_load_and_validate_upload_success(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _make_metadata(rows=24)
    frame = pd.DataFrame({"FundA": np.linspace(0.0, 1.0, metadata.rows)})
    validated = ValidatedMarketData(frame, metadata)

    monkeypatch.setattr(
        validators,
        "_read_uploaded_file",
        lambda uploaded: (frame, "source.csv"),
    )
    monkeypatch.setattr(
        validators,
        "validate_market_data",
        lambda df, source=None: validated,
    )

    attached: list[tuple[pd.DataFrame, MarketDataMetadata]] = []

    def record_attach(target: pd.DataFrame, meta: MarketDataMetadata) -> None:
        attached.append((target, meta))

    monkeypatch.setattr(validators, "attach_metadata", record_attach)

    loaded_frame, payload = validators.load_and_validate_upload(
        SimpleNamespace(name="upload.csv")
    )

    assert loaded_frame is frame
    assert attached == [(frame, metadata)]
    assert payload["metadata"] is metadata
    assert payload["validation"].is_valid
    assert payload["mode"] == metadata.mode.value
    assert payload["frequency"] == metadata.frequency_label
    assert payload["date_range"] == metadata.date_range


def test_load_and_validate_upload_raises_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(_: pd.DataFrame, source: str | None = None) -> ValidatedMarketData:
        raise MarketDataValidationError("invalid", issues=["Bad data"])

    monkeypatch.setattr(
        validators,
        "_read_uploaded_file",
        lambda uploaded: (pd.DataFrame(), "upload.csv"),
    )
    monkeypatch.setattr(validators, "validate_market_data", raise_error)

    with pytest.raises(MarketDataValidationError) as excinfo:
        validators.load_and_validate_upload(SimpleNamespace(name="upload.csv"))

    assert excinfo.value.issues == ["Bad data"]


def test_create_sample_template_structure() -> None:
    template = validators.create_sample_template()

    assert list(template.columns)[0] == "Date"
    assert template.shape == (12, 7)
    assert template["Fund_01"].dtype == np.float64
