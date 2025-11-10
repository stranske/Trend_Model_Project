"""Focused coverage tests for trend_analysis.data utilities."""

from __future__ import annotations

import io
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pandas as pd
import pytest

from trend_analysis import data as data_mod
from trend_analysis.data import (
    DEFAULT_POLICY_FALLBACK,
    _coerce_limit_entry,
    _coerce_limit_kwarg,
    _coerce_policy_kwarg,
    _finalise_validated_frame,
    _is_readable,
    _normalise_numeric_strings,
    _normalise_policy_alias,
    _validate_payload,
    ensure_datetime,
    identify_risk_free_fund,
    load_csv,
    load_parquet,
    validate_dataframe,
)
from trend_analysis.io.market_data import MarketDataValidationError, ValidatedMarketData


class DummyMetadata(SimpleNamespace):
    """Simple metadata stub that mimics the attributes used by _finalise."""

    def __init__(self, columns: list[str]) -> None:
        super().__init__(
            mode=SimpleNamespace(value="strict"),
            frequency="daily",
            frequency_detected="D",
            frequency_label="Daily",
            frequency_median_spacing_days=1.0,
            frequency_missing_periods=0,
            frequency_max_gap_periods=1,
            frequency_tolerance_periods=1,
            columns=columns,
            rows=2,
            date_range=("2020-01-01", "2020-01-02"),
            missing_policy="drop",
            missing_policy_limit=None,
            missing_policy_summary="All good",
        )


class ValidationRecorder:
    """Callable stub that records the parameters provided to validate."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.calls: list[Dict[str, Any]] = []
        self.frame = frame or pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=2), "Value": [1.0, 2.0]}).set_index(
            "Date"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> ValidatedMarketData:
        self.calls.append({"args": args, "kwargs": kwargs})
        return ValidatedMarketData(frame=self.frame, metadata=DummyMetadata(columns=["Value"]))


@pytest.fixture(autouse=True)
def restore_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure validate_market_data is restored after each test."""

    original = data_mod.validate_market_data
    yield
    monkeypatch.setattr(data_mod, "validate_market_data", original)


def test_normalise_policy_alias_variants() -> None:
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("  ") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("BackFill") == "ffill"
    assert _normalise_policy_alias(" zeros ") == "zero"
    assert _normalise_policy_alias("keep") == "keep"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("none", None),
        (5, 5),
        ("7", 7),
    ],
)
def test_coerce_limit_entry_valid(value: Any, expected: int | None) -> None:
    assert _coerce_limit_entry(value) == expected


@pytest.mark.parametrize("bad_value", [object(), "abc", -1])
def test_coerce_limit_entry_rejects_invalid_values(bad_value: Any) -> None:
    with pytest.raises(ValueError):
        _coerce_limit_entry(bad_value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("direct", "direct"),
        ({"A": "keep"}, {"A": "keep"}),
    ],
)
def test_coerce_policy_kwarg_valid(value: Any, expected: Any) -> None:
    assert _coerce_policy_kwarg(value) == expected


def test_coerce_policy_kwarg_invalid_type() -> None:
    with pytest.raises(TypeError):
        _coerce_policy_kwarg(123)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ({"A": 3}, {"A": 3}),
        (3.0, 3),
        (" 10 ", 10),
        ("none", None),
    ],
)
def test_coerce_limit_kwarg_valid(value: Any, expected: Any) -> None:
    assert _coerce_limit_kwarg(value) == expected


def test_coerce_limit_kwarg_invalid() -> None:
    with pytest.raises(TypeError):
        _coerce_limit_kwarg(object())


def test_coerce_limit_kwarg_rejects_non_numeric_string() -> None:
    with pytest.raises(TypeError):
        _coerce_limit_kwarg("abc")


def test_finalise_validated_frame_assigns_metadata_attrs() -> None:
    frame = pd.DataFrame({"Value": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    frame.index.name = "Date"
    validated = ValidatedMarketData(frame=frame, metadata=DummyMetadata(columns=["Value"]))

    result_with_dates = _finalise_validated_frame(validated, include_date_column=True)
    assert list(result_with_dates.columns) == ["Date", "Value"]
    assert result_with_dates.attrs["market_data"]["metadata"] is validated.metadata
    assert result_with_dates.attrs["market_data_mode"] == "strict"
    assert result_with_dates.attrs["market_data_columns"] == ["Value"]

    result_no_dates = _finalise_validated_frame(validated, include_date_column=False)
    assert "Date" not in result_no_dates.columns
    assert result_no_dates.attrs["market_data_rows"] == 2


def test_normalise_numeric_strings_handles_percentages_and_parentheses() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Rate": ["12%", "(3.5%)"],
            "Amount": [" 1,200 ", "800"],
        }
    )
    cleaned = _normalise_numeric_strings(frame)
    assert pytest.approx(cleaned["Rate"].iloc[0]) == 0.12
    assert pytest.approx(cleaned["Rate"].iloc[1]) == -0.035
    assert cleaned["Amount"].tolist() == [1200.0, 800.0]


def test_validate_payload_applies_policy_and_limit_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)

    payload = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Value": ["10", "20"]})

    result = _validate_payload(
        payload,
        origin="sample.csv",
        errors="log",
        include_date_column=False,
        missing_policy={"Value": "BackFill", "*": None},
        missing_limit={"Value": "5", "*": "none"},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.attrs["market_data_mode"] == "strict"

    call_kwargs = recorder.calls[0]["kwargs"]
    assert call_kwargs["missing_policy"] == {"Value": "ffill", "*": DEFAULT_POLICY_FALLBACK}
    assert call_kwargs["missing_limit"] == {"Value": 5, "*": None}


def test_validate_payload_supports_scalar_args(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)

    payload = pd.DataFrame({"Date": ["2020-01-01"], "Value": ["30"]})

    _validate_payload(
        payload,
        origin="payload",
        errors="log",
        include_date_column=True,
        missing_policy=" zeros ",
        missing_limit="7",
    )

    call_kwargs = recorder.calls[0]["kwargs"]
    assert call_kwargs["missing_policy"] == "zero"
    assert call_kwargs["missing_limit"] == 7


def test_validate_payload_policy_mapping_converts_non_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)

    payload = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Value": [1, 2]})

    _validate_payload(
        payload,
        origin="payload",
        errors="log",
        include_date_column=True,
        missing_policy={"Value": 1, "*": None},
        missing_limit={"Value": 2.0, "*": "none"},
    )

    call_kwargs = recorder.calls[0]["kwargs"]
    assert call_kwargs["missing_policy"]["Value"] == "1"
    assert call_kwargs["missing_policy"]["*"] == DEFAULT_POLICY_FALLBACK
    assert call_kwargs["missing_limit"] == {"Value": 2, "*": None}


def test_validate_payload_logs_and_returns_none(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR", "trend_analysis.data")

    def raising_validator(*_args: Any, **_kwargs: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data_mod, "validate_market_data", raising_validator)
    payload = pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]})

    result = _validate_payload(
        payload,
        origin="payload",
        errors="log",
        include_date_column=True,
    )

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_payload_propagates_errors_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_validator(*_args: Any, **_kwargs: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("boom")

    monkeypatch.setattr(data_mod, "validate_market_data", raising_validator)

    with pytest.raises(MarketDataValidationError):
        _validate_payload(
            pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]}),
            origin="payload",
            errors="raise",
            include_date_column=True,
        )


def test_is_readable_evaluates_permission_bits() -> None:
    readable = stat.S_IRUSR | stat.S_IWUSR
    unreadable = stat.S_IWUSR
    assert _is_readable(readable) is True
    assert _is_readable(unreadable) is False


def test_load_csv_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,Value\n2020-01-01,10\n2020-01-02,20\n")

    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)
    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: True)

    result = load_csv(str(csv_path), nan_policy="backFill", nan_limit="3")

    assert isinstance(result, pd.DataFrame)
    assert recorder.calls[0]["kwargs"]["source"] == str(csv_path)
    assert recorder.calls[0]["kwargs"]["missing_policy"] == "ffill"
    assert recorder.calls[0]["kwargs"]["missing_limit"] == 3


def test_load_csv_handles_missing_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "missing.csv"
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_csv(str(missing)) is None
    assert str(missing) in caplog.text


def test_load_csv_directory_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_csv(str(directory)) is None
    assert str(directory) in caplog.text


def test_load_csv_parser_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Date,Value\n1,2\n")

    def raise_parser(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad")

    monkeypatch.setattr(pd, "read_csv", raise_parser)
    caplog.set_level("ERROR", "trend_analysis.data")

    assert load_csv(str(csv_path)) is None
    assert "bad" in caplog.text


def test_load_csv_permission_denied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "restricted.csv"
    csv_path.write_text("Date,Value\n2020-01-01,1\n")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)

    with pytest.raises(PermissionError):
        load_csv(str(csv_path), errors="raise")

    # Logged error when not raising
    assert load_csv(str(csv_path), errors="log") is None


def test_load_csv_handles_validation_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("Date,Value\n2020-01-01,x\n")

    def raising_validator(*_args: Any, **_kwargs: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Unable to parse input")

    monkeypatch.setattr(data_mod, "validate_market_data", raising_validator)
    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: True)
    caplog.set_level("ERROR", "trend_analysis.data")

    assert load_csv(str(csv_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"parquet")

    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)
    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: True)
    monkeypatch.setattr(pd, "read_parquet", lambda _p: pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]}))

    result = load_parquet(str(parquet_path), nan_limit={"Value": "5"})
    assert isinstance(result, pd.DataFrame)
    assert recorder.calls[0]["kwargs"]["missing_limit"] == {"Value": 5}


def test_load_parquet_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "no_read.parquet"
    parquet_path.write_bytes(b"data")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)

    with pytest.raises(PermissionError):
        load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_permission_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    parquet_path = tmp_path / "no_read_log.parquet"
    parquet_path.write_bytes(b"data")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)
    caplog.set_level("ERROR", "trend_analysis.data")

    assert load_parquet(str(parquet_path), errors="log") is None
    assert "Permission denied" in caplog.text


def test_load_parquet_missing_file_logs(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_parquet(str(missing)) is None
    assert str(missing) in caplog.text


def test_load_parquet_validation_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    parquet_path = tmp_path / "invalid.parquet"
    parquet_path.write_bytes(b"data")

    monkeypatch.setattr(pd, "read_parquet", lambda _path: pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]}))

    def raising_validator(*_args: Any, **_kwargs: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data_mod, "validate_market_data", raising_validator)
    caplog.set_level("ERROR", "trend_analysis.data")

    assert load_parquet(str(parquet_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_dataframe_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = ValidationRecorder()
    monkeypatch.setattr(data_mod, "validate_market_data", recorder)

    df = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Value": [1, 2]})
    validate_dataframe(df, errors="log", include_date_column=False)

    assert recorder.calls[0]["kwargs"]["source"] == "dataframe"


def test_identify_risk_free_fund() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3),
            "FundA": [1.0, 1.2, 1.4],
            "FundB": [1.0, 1.0, 1.1],
            "Label": ["x", "y", "z"],
        }
    )
    assert identify_risk_free_fund(df) == "FundB"
    assert identify_risk_free_fund(pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=2)})) is None


def test_ensure_datetime_converts_and_reports_errors(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame({"Date": ["01/02/20", "01/03/20"]})
    converted = ensure_datetime(frame.copy())
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])

    bad = pd.DataFrame({"Date": ["bad", "01/03/20"]})
    caplog.set_level("ERROR", "trend_analysis.data")
    with pytest.raises(ValueError):
        ensure_datetime(bad)
    assert "malformed" in caplog.text.lower()

    iso = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"]})
    fallback = ensure_datetime(iso.copy())
    assert pd.api.types.is_datetime64_any_dtype(fallback["Date"])


