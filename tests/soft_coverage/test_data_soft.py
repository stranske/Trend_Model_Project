import logging
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

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
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    ValidatedMarketData,
)


@pytest.fixture()
def sample_metadata() -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="daily",
        frequency_detected="D",
        frequency_label="Daily",
        frequency_median_spacing_days=1.0,
        frequency_missing_periods=0,
        frequency_max_gap_periods=0,
        frequency_tolerance_periods=0,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        rows=3,
        columns=["FundA", "FundB"],
        symbols=["FundA", "FundB"],
        missing_policy="drop",
        missing_policy_limit=2,
        missing_policy_overrides={"FundA": "drop"},
        missing_policy_limits={"FundA": 2},
        missing_policy_summary="ok",
    )


def test_normalise_policy_alias_variants() -> None:
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("  ") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("both") == "ffill"
    assert _normalise_policy_alias("Backfill") == "ffill"
    assert _normalise_policy_alias("zeros") == "zero"
    assert _normalise_policy_alias("Custom") == "custom"


def test_coerce_limit_entry_handles_values() -> None:
    assert _coerce_limit_entry(None) is None
    assert _coerce_limit_entry("none") is None
    assert _coerce_limit_entry(5) == 5
    assert _coerce_limit_entry("10") == 10
    with pytest.raises(ValueError):
        _coerce_limit_entry(-1)
    with pytest.raises(ValueError):
        _coerce_limit_entry("not-int")


def test_coerce_policy_kwarg_accepts_valid_types() -> None:
    mapping = {"Fund": "drop"}
    assert _coerce_policy_kwarg(None) is None
    assert _coerce_policy_kwarg("drop") == "drop"
    assert _coerce_policy_kwarg(mapping) is mapping
    with pytest.raises(TypeError):
        _coerce_policy_kwarg(["drop"])  # type: ignore[arg-type]


def test_coerce_limit_kwarg_accepts_valid_types() -> None:
    mapping = {"FundA": "5", "FundB": None}
    assert _coerce_limit_kwarg(None) is None
    assert _coerce_limit_kwarg(mapping) == mapping
    assert _coerce_limit_kwarg(7) == 7
    assert _coerce_limit_kwarg(7.0) == 7
    assert _coerce_limit_kwarg("9") == 9
    assert _coerce_limit_kwarg("None") is None
    with pytest.raises(TypeError):
        _coerce_limit_kwarg("abc")
    with pytest.raises(TypeError):
        _coerce_limit_kwarg(["bad"])  # type: ignore[arg-type]


def test_finalise_validated_frame_includes_metadata(
    sample_metadata: MarketDataMetadata,
) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "FundA": [1.0, 2.0, 3.0],
        }
    ).set_index("Date")
    validated = ValidatedMarketData(frame=frame, metadata=sample_metadata)

    result = _finalise_validated_frame(validated, include_date_column=True)

    assert list(result.columns) == ["Date", "FundA"]
    assert "market_data" in result.attrs
    attrs = result.attrs["market_data"]
    assert attrs["metadata"] is sample_metadata
    assert result.attrs["market_data_mode"] == sample_metadata.mode.value
    assert (
        result.attrs["market_data_frequency_label"] == sample_metadata.frequency_label
    )


def test_finalise_validated_frame_without_date_column(
    sample_metadata: MarketDataMetadata,
) -> None:
    frame = pd.DataFrame(
        {"FundA": [1.0, 2.0, 3.0]}, index=pd.Index([1, 2, 3], name="Date")
    )
    validated = ValidatedMarketData(frame=frame, metadata=sample_metadata)

    result = _finalise_validated_frame(validated, include_date_column=False)

    assert list(result.index) == [1, 2, 3]
    assert result.attrs["market_data_rows"] == sample_metadata.rows


def test_normalise_numeric_strings_handles_percentages_and_parentheses() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "Return": ["10%", " (5%) ", "0.5"],
            "PnL": ["1,000", "(200)", " 300 "],
            "AlreadyNumeric": [1.0, 2.0, 3.0],
            "Text": ["alpha", "beta", "gamma"],
        }
    )

    cleaned = _normalise_numeric_strings(df)

    assert cleaned["Return"].tolist() == [0.10, -0.05, 0.005]
    assert cleaned["PnL"].tolist() == [1000.0, -200.0, 300.0]
    assert cleaned["AlreadyNumeric"].tolist() == [1.0, 2.0, 3.0]
    assert cleaned["Text"].tolist() == ["alpha", "beta", "gamma"]


def test_validate_payload_normalises_inputs(
    monkeypatch: pytest.MonkeyPatch, sample_metadata: MarketDataMetadata
) -> None:
    payload = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "FundA": ["1,000", "(200)", "50%"],
            "FundB": ["3", "4", "5"],
        }
    )
    captured: Dict[str, Any] = {}

    def fake_validate(
        df: pd.DataFrame,
        *,
        source: str,
        missing_policy: Dict[str, str] | str | None,
        missing_limit: Dict[str, int | None] | int | None,
    ) -> ValidatedMarketData:
        captured["df"] = df.copy()
        captured["source"] = source
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        normalised = df.set_index("Date")
        return ValidatedMarketData(frame=normalised, metadata=sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = _validate_payload(
        payload,
        origin="upload.csv",
        errors="log",
        include_date_column=True,
        missing_policy={"FundA": "BOTH", "FundB": None},
        missing_limit={"FundA": "7", "FundB": "none", "*": 2},
    )

    assert isinstance(result, pd.DataFrame)
    assert result["FundA"].tolist() == [10.0, -2.0, 0.5]
    assert captured["source"] == "upload.csv"
    assert captured["policy"] == {
        "FundA": "ffill",
        "FundB": DEFAULT_POLICY_FALLBACK,
    }
    assert captured["limit"] == {"FundA": 7, "FundB": None, "*": 2}
    assert result.attrs["market_data_missing_policy"] == sample_metadata.missing_policy


def test_validate_payload_handles_custom_policy_mapping(
    monkeypatch: pytest.MonkeyPatch, sample_metadata: MarketDataMetadata
) -> None:
    class StarMapping(dict):
        def __contains__(self, key: object) -> bool:  # pragma: no cover - exercised
            return key == "*" or super().__contains__(key)

    payload = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "FundA": ["1", "2"],
        }
    )
    mapping = StarMapping({"FundA": 1})
    captured: Dict[str, Any] = {}

    def fake_validate(df: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        captured.update(kwargs)
        return ValidatedMarketData(frame=df.set_index("Date"), metadata=sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    _validate_payload(
        payload,
        origin="upload.csv",
        errors="log",
        include_date_column=True,
        missing_policy=mapping,
    )

    assert captured["missing_policy"] == {"FundA": "1", "*": DEFAULT_POLICY_FALLBACK}


def test_validate_payload_logs_non_parse_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    payload = pd.DataFrame({"Date": ["bad"], "Value": [1]})

    def fake_validate(payload: pd.DataFrame, **_: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Upstream failure")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    with caplog.at_level(logging.ERROR):
        result = _validate_payload(
            payload, origin="sample.csv", errors="log", include_date_column=True
        )

    assert result is None
    assert "Upstream failure" in caplog.text


def test_validate_payload_logs_parse_failures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    payload = pd.DataFrame({"Date": ["bad"], "Value": [1]})

    def fake_validate(payload: pd.DataFrame, **_: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    with caplog.at_level(logging.ERROR):
        result = _validate_payload(
            payload, origin="sample.csv", errors="log", include_date_column=True
        )

    assert result is None
    assert "Unable to parse Date values in sample.csv" in caplog.text


def test_validate_payload_reraises_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = pd.DataFrame({"Date": ["bad"], "Value": [1]})

    def fake_validate(payload: pd.DataFrame, **_: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("bad data")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    with pytest.raises(MarketDataValidationError):
        _validate_payload(
            payload, origin="sample.csv", errors="raise", include_date_column=True
        )


def test_is_readable_checks_permission_bits() -> None:
    readable_mode = stat.S_IRUSR | stat.S_IWUSR
    not_readable_mode = stat.S_IWUSR
    assert _is_readable(readable_mode) is True
    assert _is_readable(not_readable_mode) is False


def test_load_csv_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_metadata: MarketDataMetadata
) -> None:
    csv_path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "FundA": ["1", "2"],
        }
    )
    frame.to_csv(csv_path, index=False)

    def fake_validate(df: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        assert kwargs["source"] == str(csv_path)
        return ValidatedMarketData(frame=df.set_index("Date"), metadata=sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = load_csv(
        str(csv_path),
        nan_policy="Both",
        nan_limit="4",
    )

    assert isinstance(result, pd.DataFrame)
    assert result["FundA"].tolist() == [1.0, 2.0]
    assert result.attrs["market_data_frequency"] == sample_metadata.frequency


def test_load_csv_missing_file_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    missing_path = "not_there.csv"
    with caplog.at_level(logging.ERROR):
        result = load_csv(missing_path)
    assert result is None
    assert missing_path in caplog.text


def test_load_csv_directory_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "nested"
    directory.mkdir()
    with caplog.at_level(logging.ERROR):
        result = load_csv(str(directory))
    assert result is None
    assert str(directory) in caplog.text


def test_load_csv_permission_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,FundA\n2024-01-01,1\n")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda *_: False)

    with caplog.at_level(logging.ERROR):
        result = load_csv(str(csv_path))
    assert result is None
    assert "Permission denied" in caplog.text

    with pytest.raises(PermissionError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_reports_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,FundA\n2024-01-01,1\n")

    def fake_validate_payload(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise MarketDataValidationError("Could not be parsed correctly")

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate_payload)

    with caplog.at_level(logging.ERROR):
        result = load_csv(str(csv_path))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_metadata: MarketDataMetadata
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "FundA": [1.0, 2.0],
        }
    )

    def fake_read_parquet(path: str) -> pd.DataFrame:
        assert path == str(parquet_path)
        return frame.copy()

    def fake_validate(df: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        return ValidatedMarketData(frame=df.set_index("Date"), metadata=sample_metadata)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = load_parquet(
        str(parquet_path),
        include_date_column=False,
        nan_policy="Backfill",
        nan_limit="5",
    )
    assert list(result.index) == list(frame.set_index("Date").index)


def test_load_parquet_logs_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "missing.parquet"
    parquet_path.write_bytes(b"")

    def fake_read_parquet(path: str) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("empty")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with caplog.at_level(logging.ERROR):
        result = load_parquet(str(parquet_path))
    assert result is None
    assert "empty" in caplog.text


def test_load_parquet_missing_file_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = load_parquet("/tmp/nonexistent.parquet")
    assert result is None
    assert "nonexistent.parquet" in caplog.text


def test_load_parquet_validation_error_logs_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")

    def fake_read_parquet(path: str) -> pd.DataFrame:
        return pd.DataFrame({"Date": ["2024-01-01"], "FundA": [1.0]})

    def fake_validate_payload(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise MarketDataValidationError("Other failure")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate_payload)

    with caplog.at_level(logging.ERROR):
        result = load_parquet(str(parquet_path))

    assert result is None
    assert "Other failure" in caplog.text


def test_load_parquet_logs_errors_in_raise_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "empty.parquet"
    parquet_path.write_bytes(b"")

    def fake_read_parquet(path: str) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("empty")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(pd.errors.EmptyDataError):
        load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_directory_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    with caplog.at_level(logging.ERROR):
        result = load_parquet(str(directory))
    assert result is None
    assert str(directory) in caplog.text


def test_load_parquet_permission_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda *_: False)

    with pytest.raises(PermissionError):
        load_parquet(str(parquet_path), errors="raise")


def test_validate_dataframe_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "Fund": [1.0]})
    called: Dict[str, Any] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        called["payload"] = payload
        return ValidatedMarketData(
            frame=payload.set_index("Date"),
            metadata=MarketDataMetadata(
                mode=MarketDataMode.RETURNS,
                frequency="daily",
                frequency_detected="D",
                frequency_label="Daily",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1),
                rows=1,
                columns=["Fund"],
            ),
        )

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = validate_dataframe(df, include_date_column=False, origin="frame")
    assert called["payload"].equals(df)
    assert isinstance(result, pd.DataFrame)
    assert "Fund" in result.columns


def test_identify_risk_free_fund_returns_lowest_volatility() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "FundA": [0.1, 0.1, 0.1],
            "FundB": [0.1, 0.2, 0.3],
        }
    )
    assert identify_risk_free_fund(df) == "FundA"


def test_identify_risk_free_fund_handles_missing_numeric() -> None:
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2)})
    assert identify_risk_free_fund(df) is None


def test_ensure_datetime_converts_string_dates() -> None:
    df = pd.DataFrame({"Date": ["01/02/24", "02/03/24"]})
    result = ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_raises_on_malformed(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"Date": ["2024-01-01", "bad-date"]})
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            ensure_datetime(df)
    assert "malformed date(s)" in caplog.text


def test_ensure_datetime_noop_for_existing_datetime() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"Date": dates})
    result = ensure_datetime(df)
    assert result["Date"].equals(pd.Series(dates, name="Date"))


def test_ensure_datetime_generically_parses_strings() -> None:
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"]})
    result = ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
