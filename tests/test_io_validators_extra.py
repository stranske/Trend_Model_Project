"""Additional coverage for ``trend_analysis.io.validators`` helpers."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    MissingPolicyFillDetails,
    ValidatedMarketData,
)
from trend_analysis.io.validators import (
    ValidationResult,
    _build_result,
    _read_uploaded_file,
    _ValidationSummary,
    create_sample_template,
    detect_frequency,
    validate_returns_schema,
    load_and_validate_upload,
)


def _metadata_with_warnings() -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_detected="M",
        frequency_label="Monthly",
        frequency_median_spacing_days=30.0,
        frequency_missing_periods=2,
        frequency_max_gap_periods=5,
        start=datetime(2024, 1, 31),
        end=datetime(2024, 10, 31),
        rows=10,
        columns=["FundA", "FundB"],
        missing_policy="drop",
        missing_policy_overrides={"FundA": "ffill"},
        missing_policy_limits={"FundA": 2},
        missing_policy_filled={
            "FundA": MissingPolicyFillDetails(method="ffill", count=3)
        },
        missing_policy_dropped=["FundC"],
        missing_policy_summary="ffill applied to FundA; FundC dropped",
    )


def _metadata_without_warnings() -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_detected="M",
        frequency_label="Monthly",
        frequency_median_spacing_days=30.0,
        frequency_missing_periods=0,
        frequency_max_gap_periods=0,
        start=datetime(2023, 1, 31),
        end=datetime(2024, 12, 31),
        rows=24,
        columns=["FundA", "FundB"],
        missing_policy="forward_fill",
        missing_policy_overrides={},
        missing_policy_limits={},
        missing_policy_filled={},
        missing_policy_dropped=[],
        missing_policy_summary="forward fill applied",
    )


def test_validation_summary_reports_all_warnings() -> None:
    metadata = _metadata_with_warnings()
    frame = pd.DataFrame(
        {
            "FundA": [1.0] + [None] * 9,
            "FundB": [0.01] * 10,
        }
    )
    summary = _ValidationSummary(metadata, frame)
    warnings = summary.warnings()
    assert any("small" in warning.lower() for warning in warnings)
    assert any("50% missing" in warning for warning in warnings)
    assert any("missing monthly periods" in warning.lower() for warning in warnings)
    assert any("policy dropped" in warning.lower() for warning in warnings)
    assert any("policy applied" in warning.lower() for warning in warnings)


def test_build_result_propagates_metadata() -> None:
    metadata = _metadata_with_warnings()
    frame = pd.DataFrame({"FundA": [0.01] * metadata.rows})
    validated = ValidatedMarketData(frame=frame, metadata=metadata)
    result = _build_result(validated)
    assert isinstance(result, ValidationResult)
    assert result.metadata is metadata
    assert result.warnings  # populated by summary helper


def test_validation_summary_handles_clean_dataset() -> None:
    metadata = _metadata_without_warnings()
    frame = pd.DataFrame(
        {"FundA": [0.01] * metadata.rows, "FundB": [0.02] * metadata.rows}
    )
    summary = _ValidationSummary(metadata, frame)
    assert summary.warnings() == []


@pytest.mark.parametrize("suffix", [".csv", ".xlsx", ".parquet"])
def test_read_uploaded_file_handles_path_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path = (tmp_path / "uploads").resolve()
    path.mkdir(exist_ok=True)
    path = path / f"data{suffix}"
    frame = pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.01]})
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".xlsx":
        path.write_bytes(b"")
        monkeypatch.setattr(pd, "read_excel", lambda _path: frame)
    else:
        path.write_bytes(b"")
        monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    loaded, source = _read_uploaded_file(str(path))
    assert isinstance(loaded, pd.DataFrame)
    assert source.endswith(suffix)


def test_read_uploaded_file_missing_and_directory_paths(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.csv"
    with pytest.raises(ValueError, match="File not found"):
        _read_uploaded_file(str(missing_path))

    directory_path = tmp_path / "dir"
    directory_path.mkdir()
    with pytest.raises(ValueError, match="directory"):
        _read_uploaded_file(str(directory_path))


def test_read_uploaded_file_stream_variants() -> None:
    frame = pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.02]})
    buf = io.BytesIO()
    frame.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "upload.xlsx"

    loaded, source = _read_uploaded_file(buf)
    assert loaded.shape == frame.shape
    assert source == "upload.xlsx"


def test_read_uploaded_file_parquet_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO(b"parquet")
    buf.name = "upload.parquet"
    monkeypatch.setattr(pd, "read_parquet", lambda _buf: pd.DataFrame({"A": [1]}))
    loaded, source = _read_uploaded_file(buf)
    assert source == "upload.parquet"
    assert list(loaded.columns) == ["A"]


def test_read_uploaded_file_fallback_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_like = SimpleNamespace(name="mystery.csv")

    def fake_read_csv(obj: Any) -> pd.DataFrame:
        if isinstance(obj, SimpleNamespace):
            return pd.DataFrame({"Date": ["2024-01-31"]})
        raise pd.errors.ParserError("bad data")

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    loaded, source = _read_uploaded_file(csv_like)
    assert source == "mystery.csv"
    assert "Date" in loaded.columns

    class Broken(io.BytesIO):
        name = "broken.csv"

        def read(self, *args: Any, **kwargs: Any) -> bytes:  # type: ignore[override]
            raise pd.errors.ParserError("bad data")

    broken = Broken()
    with pytest.raises(ValueError, match="Failed to parse"):
        _read_uploaded_file(broken)

    with pytest.raises(ValueError, match="Unsupported upload source"):
        _read_uploaded_file(object())


def test_read_uploaded_file_stream_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.BytesIO(b"payload")
    stream.name = "upload.csv"

    def explode(_obj: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", explode)
    with pytest.raises(ValueError, match="Failed to read file"):
        _read_uploaded_file(stream)


def test_read_uploaded_file_lower_name_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_like = SimpleNamespace(name="broken.csv")

    def explode(_obj: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad csv")

    monkeypatch.setattr(pd, "read_csv", explode)
    with pytest.raises(ValueError, match="Failed to parse"):
        _read_uploaded_file(file_like)


@pytest.mark.parametrize(
    "exc,msg",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "directory"),
        (pd.errors.EmptyDataError("empty"), "contains no data"),
    ],
)
def test_read_uploaded_file_propagates_file_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, msg: str
) -> None:
    stream = io.BytesIO()
    stream.name = "broken.csv"

    def raise_exc(_obj: Any) -> pd.DataFrame:
        raise exc

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError, match=msg):
        _read_uploaded_file(stream)


def test_read_uploaded_file_path_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("bad")

    def boom(_path: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", boom)
    with pytest.raises(ValueError, match="Failed to read"):
        _read_uploaded_file(str(csv_path))


@pytest.mark.parametrize(
    "exc,msg",
    [
        (FileNotFoundError("missing"), "File not found"),
        (PermissionError("denied"), "Permission denied"),
        (IsADirectoryError("dir"), "directory"),
        (pd.errors.EmptyDataError("empty"), "contains no data"),
        (RuntimeError("boom"), "Failed to read"),
    ],
)
def test_read_uploaded_file_lower_name_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, msg: str
) -> None:
    stub = SimpleNamespace(name="fallback.csv")

    def raise_exc(_obj: Any) -> pd.DataFrame:
        raise exc

    monkeypatch.setattr(pd, "read_csv", raise_exc)
    with pytest.raises(ValueError, match=msg):
        _read_uploaded_file(stub)


def test_load_and_validate_upload_returns_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata_with_warnings()
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-31", periods=metadata.rows, freq="M"),
            "FundA": 0.01,
        }
    )

    monkeypatch.setattr(
        "trend_analysis.io.validators._read_uploaded_file",
        lambda *_args, **_kwargs: (frame, "uploaded.csv"),
    )

    def fake_validate(data: pd.DataFrame, source: str):
        assert source == "uploaded.csv"
        return ValidatedMarketData(frame=data.set_index("Date"), metadata=metadata)

    monkeypatch.setattr(
        "trend_analysis.io.validators.validate_market_data", fake_validate
    )

    loaded_frame, meta = load_and_validate_upload("dummy")
    assert isinstance(meta["validation"], ValidationResult)
    assert meta["metadata"] is metadata
    assert meta["mode"] == metadata.mode.value
    assert list(meta["original_columns"]) == metadata.columns
    assert loaded_frame.index.name == "Date"


def test_load_and_validate_upload_reraises_market_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.01]})
    csv = io.StringIO()
    frame.to_csv(csv, index=False)
    csv.seek(0)
    csv.name = "data.csv"

    error = MarketDataValidationError("invalid data", issues=["bad"])

    monkeypatch.setattr(
        "trend_analysis.io.validators._read_uploaded_file",
        lambda *_args, **_kwargs: (frame, "data.csv"),
    )

    def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(
        "trend_analysis.io.validators.validate_market_data", raise_error
    )

    with pytest.raises(MarketDataValidationError) as excinfo:
        load_and_validate_upload(csv)

    assert excinfo.value.issues == ["bad"]


def test_create_sample_template_has_expected_shape() -> None:
    template = create_sample_template()
    assert template.shape == (12, 7)
    assert template.columns[0] == "Date"


def test_validation_result_report_includes_metadata() -> None:
    metadata = _metadata_with_warnings()
    result = ValidationResult(
        True,
        [],
        ["be mindful"],
        frequency="Monthly",
        date_range=("2024-01-31", "2024-10-31"),
        metadata=metadata,
    )
    report = result.get_report()
    assert "✅" in report
    assert "📊 Detected frequency" in report
    assert "🧹 Missing data policy" in report

    failure = ValidationResult(False, ["broken"], [])
    assert "❌" in failure.get_report()


def test_validation_result_report_omits_optional_metadata() -> None:
    """Ensure optional fields are skipped when absent."""

    result = ValidationResult(
        True, [], [], frequency=None, date_range=None, metadata=None
    )
    report = result.get_report()
    assert "✅ Schema validation passed" in report
    assert "Detected frequency" not in report
    assert "Date range" not in report
    assert "Detected mode" not in report


def test_detect_frequency_handles_irregular(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="M")
    df = pd.DataFrame(index=index)

    def raise_error(_index: pd.Index) -> dict[str, str]:
        raise MarketDataValidationError("Irregular cadence detected")

    monkeypatch.setattr("trend_analysis.io.validators.classify_frequency", raise_error)
    assert "irregular" in detect_frequency(df)


def test_detect_frequency_returns_code(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="M")
    df = pd.DataFrame(index=index)
    monkeypatch.setattr(
        "trend_analysis.io.validators.classify_frequency",
        lambda _index: {"label": "unknown", "code": "M"},
    )
    assert detect_frequency(df) == "M"


def test_detect_frequency_handles_non_datetime_index() -> None:
    df = pd.DataFrame({"FundA": [0.1, 0.2]}, index=[0, 1])
    assert detect_frequency(df) == "unknown"


def test_detect_frequency_returns_label(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="M")
    df = pd.DataFrame(index=index)

    def classify(idx: pd.Index) -> Dict[str, Any]:
        assert idx is index
        return {"label": "Monthly", "code": "M"}

    monkeypatch.setattr(
        "trend_analysis.io.validators.classify_frequency", classify
    )
    assert detect_frequency(df) == "Monthly"


def test_detect_frequency_returns_unknown_on_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="ME")
    df = pd.DataFrame(index=index)

    def raise_error(_index: pd.Index) -> dict[str, str]:
        raise MarketDataValidationError("Validation failed")

    monkeypatch.setattr("trend_analysis.io.validators.classify_frequency", raise_error)
    assert detect_frequency(df) == "unknown"


def test_validate_returns_schema_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({"FundA": [0.1, 0.2]})
    error = MarketDataValidationError("boom", issues=["broken"])

    def raise_error(_df: Any) -> Any:
        raise error

    monkeypatch.setattr(
        "trend_analysis.io.validators.validate_market_data", raise_error
    )

    result = validate_returns_schema(df)
    assert not result.is_valid
    assert result.issues == ["broken"]
