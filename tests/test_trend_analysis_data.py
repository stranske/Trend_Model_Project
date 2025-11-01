from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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


@pytest.fixture
def sample_metadata() -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="B",
        frequency_label="daily",
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 3),
        rows=2,
        columns=["A", "B"],
    )


def test_normalise_policy_alias_variants():
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("  ") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("bOtH") == "ffill"
    assert _normalise_policy_alias("zeros") == "zero"
    assert _normalise_policy_alias("custom") == "custom"


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("none", None), (5, 5), ("6", 6), (1.2, 1)],
)
def test_coerce_limit_entry_valid(value: Any, expected: int | None):
    assert _coerce_limit_entry(value) == expected


@pytest.mark.parametrize("value", ["abc", object()])
def test_coerce_limit_entry_invalid_type(value: Any):
    with pytest.raises(ValueError):
        _coerce_limit_entry(value)


def test_coerce_limit_entry_negative():
    with pytest.raises(ValueError):
        _coerce_limit_entry(-1)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("drop", "drop"),
        ({"A": "ffill"}, {"A": "ffill"}),
    ],
)
def test_coerce_policy_kwarg_valid(value: Any, expected: Any):
    assert _coerce_policy_kwarg(value) == expected


def test_coerce_policy_kwarg_invalid():
    with pytest.raises(TypeError):
        _coerce_policy_kwarg(123)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ({"A": 1}, {"A": 1}),
        (5.0, 5),
        ("7", 7),
        (" none ", None),
    ],
)
def test_coerce_limit_kwarg_valid(value: Any, expected: Any):
    assert _coerce_limit_kwarg(value) == expected


@pytest.mark.parametrize("value", [object(), "abc"])
def test_coerce_limit_kwarg_invalid(value: Any):
    with pytest.raises(TypeError):
        _coerce_limit_kwarg(value)


def test_normalise_numeric_strings_handles_percentages_and_parentheses():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Return": ["10.0%", "(5.0%)"],
            "Count": ["1,000", "2,000"],
            "AlreadyNumeric": [1.0, 2.0],
            "NonNumeric": ["abc", ""],
        }
    )
    cleaned = _normalise_numeric_strings(df)
    assert cleaned["Return"].tolist() == [0.1, -0.05]
    assert cleaned["Count"].tolist() == [1000, 2000]
    assert cleaned["AlreadyNumeric"].tolist() == [1.0, 2.0]
    assert cleaned["NonNumeric"].tolist() == ["abc", ""]


def test_finalise_validated_frame_includes_metadata(
    sample_metadata: MarketDataMetadata,
):
    frame = pd.DataFrame(
        {"A": [1, 2]}, index=pd.Index(["2020-01-01", "2020-01-02"], name="Date")
    )
    validated = ValidatedMarketData(frame=frame, metadata=sample_metadata)

    result_with_dates = _finalise_validated_frame(validated, include_date_column=True)
    assert "Date" in result_with_dates.columns
    assert result_with_dates.attrs["market_data"]["metadata"] == sample_metadata
    assert (
        result_with_dates.attrs["market_data_missing_policy"]
        == sample_metadata.missing_policy
    )

    result_without_dates = _finalise_validated_frame(
        validated, include_date_column=False
    )
    assert "Date" not in result_without_dates.columns


def test_validate_payload_normalises_inputs(
    monkeypatch, sample_metadata: MarketDataMetadata
):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": ["1.0"]})

    def fake_validate(df, *, source, missing_policy, missing_limit):
        expected = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)
        assert source == "source.csv"
        assert missing_policy == {"A": "ffill", "*": DEFAULT_POLICY_FALLBACK}
        assert missing_limit == {"A": 2, "*": None}
        validated_frame = pd.DataFrame(
            {"Date": pd.to_datetime(["2020-01-01"]), "A": [1.0]}
        )
        return ValidatedMarketData(validated_frame.set_index("Date"), sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = _validate_payload(
        payload,
        origin="source.csv",
        errors="log",
        include_date_column=True,
        missing_policy={"A": "both", "*": None},
        missing_limit={"A": "2", "*": "none"},
    )
    assert isinstance(result, pd.DataFrame)
    assert result.attrs["market_data_mode"] == sample_metadata.mode.value


def test_validate_payload_with_string_policy(
    monkeypatch, sample_metadata: MarketDataMetadata
):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def fake_validate(df, *, source, missing_policy, missing_limit):
        assert missing_policy == "zero"
        assert missing_limit == 10
        return ValidatedMarketData(df.set_index("Date"), sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    result = _validate_payload(
        payload,
        origin="source.csv",
        errors="log",
        include_date_column=False,
        missing_policy="Zeros",
        missing_limit="10",
    )
    assert isinstance(result, pd.DataFrame)
    assert "Date" not in result.columns


def test_validate_payload_mapping_adds_default(
    monkeypatch, sample_metadata: MarketDataMetadata
):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    class StarMapping(dict):
        def __contains__(self, key: object) -> bool:
            return True if key == "*" else super().__contains__(key)

    star_policy = StarMapping({"A": "ffill"})

    def fake_validate(df, *, missing_policy, missing_limit, **kwargs):
        assert missing_policy == {"A": "ffill", "*": DEFAULT_POLICY_FALLBACK}
        assert missing_limit is None
        return ValidatedMarketData(df.set_index("Date"), sample_metadata)

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    result = _validate_payload(
        payload,
        origin="source.csv",
        errors="log",
        include_date_column=True,
        missing_policy=star_policy,
    )
    assert isinstance(result, pd.DataFrame)


def test_validate_payload_handles_validation_error(monkeypatch, caplog):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def raise_error(*args, **kwargs):
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", raise_error)
    with caplog.at_level("ERROR"):
        assert (
            _validate_payload(
                payload, origin="f", errors="log", include_date_column=True
            )
            is None
        )
    assert any("Unable to parse Date values" in message for message in caplog.messages)


def test_validate_payload_logs_generic_error(monkeypatch, caplog):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def raise_error(*args, **kwargs):
        raise MarketDataValidationError("Other failure")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", raise_error)
    with caplog.at_level("ERROR"):
        assert (
            _validate_payload(
                payload, origin="f", errors="log", include_date_column=True
            )
            is None
        )
    assert any("Other failure" in message for message in caplog.messages)


def test_validate_payload_raises_when_requested(monkeypatch):
    payload = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def raise_error(*args, **kwargs):
        raise MarketDataValidationError("Bad data")

    monkeypatch.setattr("trend_analysis.data.validate_market_data", raise_error)
    with pytest.raises(MarketDataValidationError):
        _validate_payload(payload, origin="f", errors="raise", include_date_column=True)


@pytest.mark.parametrize(
    "mode, expected",
    [
        (0, False),
        (0o400, True),
        (0o040, True),
        (0o004, True),
    ],
)
def test_is_readable(mode: int, expected: bool):
    assert _is_readable(mode) == expected


def test_load_csv_success(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")

    def fake_validate(
        df, *, origin, errors, include_date_column, missing_policy, missing_limit
    ):
        assert origin == str(csv_path)
        assert errors == "log"
        assert include_date_column is True
        assert missing_policy == "drop"
        assert missing_limit == 3
        return df.assign(validated=True)

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)
    result = load_csv(
        str(csv_path),
        missing_policy="drop",
        missing_limit=3.0,
    )
    assert list(result.columns) == ["Date", "A", "validated"]


def test_load_csv_permission_denied(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_permission_logged(tmp_path: Path, monkeypatch, caplog):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: False)

    with caplog.at_level("ERROR"):
        assert load_csv(str(csv_path)) is None
    assert any("Permission denied" in message for message in caplog.messages)


def test_load_csv_handles_known_errors(tmp_path: Path, monkeypatch, caplog):
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text("", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda path: (_ for _ in ()).throw(pd.errors.EmptyDataError("empty")),
    )

    with caplog.at_level("ERROR"):
        assert load_csv(str(csv_path)) is None
    assert any("empty" in message for message in caplog.messages)


def test_load_csv_legacy_kwargs(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")

    def fake_validate(df, *, missing_policy, missing_limit, **kwargs):
        assert missing_policy == "backfill"
        assert missing_limit == 7
        return df

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)
    result = load_csv(
        str(csv_path),
        nan_policy="backfill",
        nan_limit="7",
    )
    assert isinstance(result, pd.DataFrame)


def test_load_csv_directory_error(tmp_path: Path, caplog):
    directory = tmp_path / "csv_dir"
    directory.mkdir()
    with caplog.at_level("ERROR"):
        assert load_csv(str(directory)) is None
    assert any(
        "Is a directory" in message or str(directory) in message
        for message in caplog.messages
    )


def test_load_csv_market_validation_error(tmp_path: Path, monkeypatch, caplog):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("Parse problem")

    monkeypatch.setattr("trend_analysis.data._validate_payload", raise_validation)
    with caplog.at_level("ERROR"):
        assert load_csv(str(csv_path)) is None
    assert any("Parse problem" in message for message in caplog.messages)


def test_load_csv_validation_error_raise(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n2020-01-01,1\n", encoding="utf-8")

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("failure")

    monkeypatch.setattr("trend_analysis.data._validate_payload", raise_validation)
    with pytest.raises(MarketDataValidationError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_unexpected_exception(tmp_path: Path, monkeypatch, caplog):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Date,A\n", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_csv", lambda path: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with caplog.at_level("ERROR"):
        assert load_csv(str(csv_path)) is None
    assert any("Unexpected error" in message for message in caplog.messages)


def test_load_parquet_success(tmp_path: Path, monkeypatch):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")

    def fake_read_parquet(path):
        assert path == str(parquet_path)
        return pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def fake_validate(df, **kwargs):
        return df.assign(validated=True)

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr("pandas.read_parquet", fake_read_parquet)
    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)

    result = load_parquet(str(parquet_path), missing_limit="5")
    assert "validated" in result.columns


def test_load_parquet_missing_file(tmp_path: Path, caplog):
    path = tmp_path / "missing.parquet"
    with caplog.at_level("ERROR"):
        assert load_parquet(str(path)) is None
    assert any(
        "No such file" in message or "missing.parquet" in message
        for message in caplog.messages
    )


def test_load_parquet_permission_error(tmp_path: Path, monkeypatch):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_legacy_kwargs(tmp_path: Path, monkeypatch):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: pd.DataFrame({"Date": ["2020-01-01"], "A": [2]}),
    )

    def fake_validate(df, *, missing_policy, missing_limit, **kwargs):
        assert missing_policy == "zeros"
        assert missing_limit == 5
        return df

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)
    result = load_parquet(
        str(parquet_path),
        nan_policy="zeros",
        nan_limit="5",
    )
    assert isinstance(result, pd.DataFrame)


def test_load_parquet_validation_error(tmp_path: Path, monkeypatch, caplog):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: pd.DataFrame({"Date": ["2020-01-01"], "A": [1]}),
    )

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("Validation failed")

    monkeypatch.setattr("trend_analysis.data._validate_payload", raise_validation)
    with caplog.at_level("ERROR"):
        assert load_parquet(str(parquet_path)) is None
    assert any("Validation failed" in message for message in caplog.messages)


def test_load_parquet_validation_error_raise(tmp_path: Path, monkeypatch):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: pd.DataFrame({"Date": ["2020-01-01"], "A": [1]}),
    )

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data._validate_payload", raise_validation)
    with pytest.raises(MarketDataValidationError):
        load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_parse_message(tmp_path: Path, monkeypatch, caplog):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr("trend_analysis.data._is_readable", lambda mode: True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: pd.DataFrame({"Date": ["2020-01-01"], "A": [1]}),
    )

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr("trend_analysis.data._validate_payload", raise_validation)
    with caplog.at_level("ERROR"):
        assert load_parquet(str(parquet_path)) is None
    assert any("Unable to parse" in message for message in caplog.messages)


def test_validate_dataframe_uses_validate_payload(monkeypatch):
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1.0]})

    def fake_validate(*args, **kwargs):
        assert kwargs["origin"] == "dataframe"
        return pd.DataFrame({"A": [1.0]})

    monkeypatch.setattr("trend_analysis.data._validate_payload", fake_validate)
    result = validate_dataframe(df)
    assert list(result.columns) == ["A"]


def test_identify_risk_free_fund_selects_lowest_vol():
    df = pd.DataFrame({"Date": [1, 2, 3], "A": [1, 1, 1], "B": [1, 2, 3]})
    assert identify_risk_free_fund(df) == "A"


def test_identify_risk_free_fund_no_numeric_columns():
    df = pd.DataFrame({"Date": ["a"], "Name": ["fund"]})
    assert identify_risk_free_fund(df) is None


def test_ensure_datetime_successful_conversion():
    df = pd.DataFrame({"Date": ["01/02/20", "02/02/20"]})
    result = ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])  # type: ignore[arg-type]


def test_ensure_datetime_detects_malformed(caplog):
    df = pd.DataFrame({"Date": ["bad", "01/02/20"]})
    with pytest.raises(ValueError):
        ensure_datetime(df.copy())
    assert any("malformed" in message.lower() for message in caplog.messages)


def test_ensure_datetime_skips_when_already_datetime():
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    result = ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])  # type: ignore[arg-type]
