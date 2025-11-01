"""Focused tests for :mod:`trend_analysis.data`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import stat
from typing import Mapping

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
from trend_analysis.io.market_data import MarketDataMetadata, MarketDataMode, ValidatedMarketData


@pytest.fixture
def sample_metadata() -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="daily",
        frequency_detected="D",
        frequency_label="Daily",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        rows=2,
        columns=["FundA", "FundB"],
        missing_policy="drop",
        missing_policy_limit=None,
        missing_policy_overrides={},
        missing_policy_limits={},
        missing_policy_filled={},
        missing_policy_dropped=[],
    )


@pytest.fixture
def validated(sample_metadata: MarketDataMetadata) -> ValidatedMarketData:
    frame = pd.DataFrame(
        {"FundA": [0.1, 0.2], "FundB": [0.3, 0.1]},
        index=pd.Index(pd.date_range("2024-01-01", periods=2, freq="D"), name="Date"),
    )
    return ValidatedMarketData(frame=frame, metadata=sample_metadata)


def test_normalise_policy_alias_variants() -> None:
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("   ") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("Both") == "ffill"
    assert _normalise_policy_alias("zeros") == "zero"
    assert _normalise_policy_alias("drop") == "drop"


def test_coerce_limit_entry_accepts_and_rejects() -> None:
    assert _coerce_limit_entry(None) is None
    assert _coerce_limit_entry("none") is None
    assert _coerce_limit_entry("10") == 10

    with pytest.raises(ValueError):
        _coerce_limit_entry("not-an-int")
    with pytest.raises(ValueError):
        _coerce_limit_entry(-1)


def test_coerce_policy_kwarg_validates_type() -> None:
    mapping: Mapping[str, str] = {"FundA": "drop"}
    assert _coerce_policy_kwarg("drop") == "drop"
    assert _coerce_policy_kwarg(mapping) == mapping
    assert _coerce_policy_kwarg(None) is None

    with pytest.raises(TypeError):
        _coerce_policy_kwarg(42)


def test_coerce_limit_kwarg_variants() -> None:
    mapping: Mapping[str, int | None] = {"FundA": 3, "FundB": None}
    assert _coerce_limit_kwarg(5) == 5
    assert _coerce_limit_kwarg(3.0) == 3
    assert _coerce_limit_kwarg("7") == 7
    assert _coerce_limit_kwarg(" none ") is None
    assert _coerce_limit_kwarg(mapping) == mapping

    with pytest.raises(TypeError):
        _coerce_limit_kwarg("bad-value")


def test_coerce_limit_kwarg_additional_paths() -> None:
    assert _coerce_limit_kwarg(None) is None
    assert _coerce_limit_kwarg({"FundA": "5"}) == {"FundA": "5"}
    assert _coerce_limit_kwarg("null") is None


def test_normalise_numeric_strings_handles_percentages() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Return": [" 1.5%", "(2.5%)"],
            "Level": ["1,200", "1,400"],
            "Ignore": ["text", ""],
        }
    )
    cleaned = _normalise_numeric_strings(frame)
    assert pytest.approx(cleaned["Return"].iloc[0]) == 0.015
    assert pytest.approx(cleaned["Return"].iloc[1]) == -0.025
    assert cleaned["Level"].tolist() == [1200, 1400]
    assert cleaned["Ignore"].tolist() == ["text", ""]


def test_finalise_validated_frame_round_trips_attrs(validated: ValidatedMarketData) -> None:
    result = _finalise_validated_frame(validated, include_date_column=True)
    assert "Date" in result.columns
    attrs = result.attrs["market_data"]
    assert attrs["metadata"].mode is MarketDataMode.RETURNS
    assert result.attrs["market_data_missing_policy"] == "drop"
    assert result.attrs["market_data_rows"] == 2


def test_validate_payload_normalises_policy_and_limit(monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert isinstance(kwargs["missing_policy"], dict)
        assert kwargs["missing_policy"]["FundA"] == "ffill"
        assert kwargs["missing_policy"]["*"] == DEFAULT_POLICY_FALLBACK
        assert isinstance(kwargs["missing_limit"], dict)
        assert kwargs["missing_limit"]["FundB"] == 5
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "FundA": [0.1, 0.2],
            "FundB": [0.3, 0.1],
        }
    )
    mapping_policy = {"FundA": "bfill", "FundB": "drop", "*": "drop"}
    mapping_limit = {"FundA": "none", "FundB": 5}

    result = _validate_payload(
        payload,
        origin="upload.csv",
        errors="raise",
        include_date_column=True,
        missing_policy=mapping_policy,
        missing_limit=mapping_limit,
    )
    assert result is not None
    assert list(result.columns) == ["Date", "FundA", "FundB"]


def test_validate_payload_scalar_policy(monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["missing_policy"] == "zero"
        assert kwargs["missing_limit"] == 10
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "FundA": [0.1, 0.2],
        }
    )

    result = _validate_payload(
        payload,
        origin="memory",
        errors="raise",
        include_date_column=False,
        missing_policy="zeros",
        missing_limit="10",
    )
    assert result is not None
    assert "Date" not in result.columns


def test_validate_payload_policy_handles_non_string(monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    class Alias:
        def __str__(self) -> str:
            return "bfill"

    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        policy = kwargs["missing_policy"]
        assert isinstance(policy, dict)
        assert policy["FundC"] == "ffill"
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "FundC": [0.5]})
    mapping_policy = {"FundC": Alias(), "FundD": None, "*": "drop"}

    result = _validate_payload(
        payload,
        origin="memory",
        errors="raise",
        include_date_column=False,
        missing_policy=mapping_policy,
    )
    assert result is not None


def test_validate_payload_logs_and_suppresses_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Generic validation failure")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "FundA": [0.1]})

    with caplog.at_level("ERROR"):
        result = _validate_payload(
            payload,
            origin="upload.csv",
            errors="log",
            include_date_column=False,
        )
    assert result is None
    assert "Generic validation failure" in caplog.text


def test_validate_payload_raises_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("serious failure")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "FundA": [0.1]})

    with pytest.raises(MarketDataValidationError):
        _validate_payload(
            payload,
            origin="upload.csv",
            errors="raise",
            include_date_column=False,
        )


def test_validate_payload_handles_parse_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed", issues=["bad date"])

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    payload = pd.DataFrame({"Date": ["invalid"], "FundA": [0.1]})

    with caplog.at_level("ERROR"):
        result = _validate_payload(
            payload,
            origin="upload.csv",
            errors="log",
            include_date_column=True,
        )
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_is_readable_checks_permission_bits() -> None:
    read_all = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    assert _is_readable(read_all)
    assert not _is_readable(0)


def test_load_csv_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.1]})
    df.to_csv(csv_path, index=False)

    def fake_validate(payload: pd.DataFrame, **_kwargs: object) -> ValidatedMarketData:
        assert payload.equals(pd.read_csv(csv_path))
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = load_csv(str(csv_path), errors="raise", include_date_column=True)
    assert result is not None
    assert "Date" in result.columns


def test_load_csv_legacy_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.2]}).to_csv(csv_path, index=False)

    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["missing_policy"] == "ffill"
        assert kwargs["missing_limit"] == 5
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = load_csv(
        str(csv_path),
        errors="raise",
        nan_policy="both",
        nan_limit="5",
        missing_limit="5",
    )
    assert result is not None


def test_load_csv_invalid_nan_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid_limit.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.2]}).to_csv(csv_path, index=False)

    with pytest.raises(TypeError):
        load_csv(str(csv_path), errors="raise", nan_limit="bad")


def test_load_csv_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "locked.csv"
    csv_path.write_text("Date,FundA\n2024-01-01,0.1\n", encoding="utf-8")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda _mode: False)
    with pytest.raises(PermissionError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_permission_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "locked.csv"
    csv_path.write_text("Date,FundA\n2024-01-01,0.1\n", encoding="utf-8")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda _mode: False)
    with caplog.at_level("ERROR"):
        result = load_csv(str(csv_path), errors="log")
    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_directory_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    directory = tmp_path / "data_dir"
    directory.mkdir()
    with caplog.at_level("ERROR"):
        result = load_csv(str(directory), errors="log")
    assert result is None
    assert "Is a directory" in caplog.text or str(directory) in caplog.text


def test_load_csv_validation_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.1]}).to_csv(csv_path, index=False)

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Unable to parse data")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    with caplog.at_level("ERROR"):
        result = load_csv(str(csv_path), errors="log")
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_missing_limit_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData
) -> None:
    csv_path = tmp_path / "limits.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.4]}).to_csv(csv_path, index=False)

    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["missing_limit"] == 4
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = load_csv(str(csv_path), errors="raise", missing_limit="4")
    assert result is not None


def test_load_csv_wrapper_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    csv_path = tmp_path / "wrapper.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.3]}).to_csv(csv_path, index=False)

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)

    with caplog.at_level("ERROR"):
        result = load_csv(str(csv_path), errors="log")
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_validation_error_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    csv_path = tmp_path / "raise.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.1]}).to_csv(csv_path, index=False)

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("failure")

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)

    with pytest.raises(MarketDataValidationError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_logs_missing_file(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fake_read_csv(_path: str, *args: object, **kwargs: object) -> pd.DataFrame:
        raise FileNotFoundError("missing.csv")

    monkeypatch.setattr("pandas.read_csv", fake_read_csv)
    with caplog.at_level("ERROR"):
        result = load_csv("missing.csv", errors="log")
    assert result is None
    assert "missing.csv" in caplog.text


def test_load_parquet_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"parquet")

    def fake_read_parquet(_path: str, *args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "FundA": [0.1]})

    def fake_validate(payload: pd.DataFrame, **_kwargs: object) -> ValidatedMarketData:
        return validated

    monkeypatch.setattr("pandas.read_parquet", fake_read_parquet)
    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = load_parquet(str(parquet_path), errors="raise", include_date_column=False)
    assert result is not None
    assert "Date" not in result.columns


def test_load_parquet_legacy_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData
) -> None:
    parquet_path = tmp_path / "legacy.parquet"
    parquet_path.write_bytes(b"parquet")

    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.2]}))

    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["missing_policy"] == "ffill"
        assert kwargs["missing_limit"] == 4
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = load_parquet(
        str(parquet_path),
        errors="raise",
        nan_policy="both",
        nan_limit=4,
    )
    assert result is not None


def test_load_parquet_invalid_nan_limit(tmp_path: Path) -> None:
    parquet_path = tmp_path / "invalid_limit.parquet"
    parquet_path.write_bytes(b"data")

    with pytest.raises(TypeError):
        load_parquet(str(parquet_path), errors="raise", nan_limit="bad")


def test_load_parquet_missing_limit_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData
) -> None:
    parquet_path = tmp_path / "limits.parquet"
    parquet_path.write_bytes(b"parquet")

    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Date": ["2024-01-01"], "FundA": [0.2]}))

    def fake_validate(payload: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["missing_limit"] == 3
        return validated

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = load_parquet(str(parquet_path), errors="raise", missing_limit="3")
    assert result is not None


def test_load_parquet_permission_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "locked.parquet"
    parquet_path.write_bytes(b"data")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda _mode: False)
    with caplog.at_level("ERROR"):
        result = load_parquet(str(parquet_path), errors="log")
    assert result is None
    assert str(parquet_path) in caplog.text


def test_load_parquet_directory_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    directory = tmp_path / "parquet_dir"
    directory.mkdir()
    with caplog.at_level("ERROR"):
        result = load_parquet(str(directory), errors="log")
    assert result is None
    assert str(directory) in caplog.text


def test_load_parquet_logs_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed", issues=["bad date"])

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: pd.DataFrame())

    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"data")

    with caplog.at_level("ERROR"):
        result = load_parquet(str(parquet_path), errors="log")
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_wrapper_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    parquet_path = tmp_path / "wrapper.parquet"
    parquet_path.write_bytes(b"data")

    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: pd.DataFrame())

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)

    with caplog.at_level("ERROR"):
        result = load_parquet(str(parquet_path), errors="log")
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_validation_error_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trend_analysis.io.market_data import MarketDataValidationError

    parquet_path = tmp_path / "raise.parquet"
    parquet_path.write_bytes(b"data")
    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: pd.DataFrame())

    def fake_validate(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("failure")

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)

    with pytest.raises(MarketDataValidationError):
        load_parquet(str(parquet_path), errors="raise")


def test_validate_dataframe_calls_validate(monkeypatch: pytest.MonkeyPatch, validated: ValidatedMarketData) -> None:
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=1), "FundA": [0.1]})

    def fake_validate(frame: pd.DataFrame, **kwargs: object) -> ValidatedMarketData:
        assert kwargs["origin"] == "dataframe"
        return validated

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)
    result = validate_dataframe(payload, errors="raise", include_date_column=False)
    assert result is not None


def test_identify_risk_free_fund_prefers_lowest_volatility() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "FundA": [0.1, 0.2, 0.15],
            "FundB": [0.05, 0.02, 0.03],
            "Text": ["a", "b", "c"],
        }
    )
    assert identify_risk_free_fund(df) == "FundB"
    assert identify_risk_free_fund(df[["Date", "Text"]]) is None


def test_ensure_datetime_successful_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"Date": ["01/01/24", "01/02/24"]})
    ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    bad = pd.DataFrame({"Date": ["01/01/24", "not-a-date"]})
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            ensure_datetime(bad)
    assert "malformed" in caplog.text.lower()


def test_ensure_datetime_skips_missing_column() -> None:
    df = pd.DataFrame({"Other": [1, 2, 3]})
    result = ensure_datetime(df, column="Date")
    assert result is df


def test_ensure_datetime_generic_parse_success() -> None:
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"]})
    ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
