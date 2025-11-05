from datetime import datetime

import pandas as pd
import pytest

from trend_analysis import data
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
        frequency="D",
        frequency_detected="D",
        frequency_label="daily",
        frequency_median_spacing_days=1.0,
        frequency_missing_periods=0,
        frequency_max_gap_periods=0,
        frequency_tolerance_periods=1,
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 2),
        rows=2,
        columns=["FundA", "FundB"],
        missing_policy="drop",
        missing_policy_summary="all clear",
    )


def test_normalise_policy_alias_handles_aliases():
    assert data._normalise_policy_alias(None) == "drop"
    assert data._normalise_policy_alias("  ") == "drop"
    assert data._normalise_policy_alias("Both") == "ffill"
    assert data._normalise_policy_alias("zeros") == "zero"
    assert data._normalise_policy_alias("custom") == "custom"


def test_coerce_limit_entry_accepts_numbers_and_rejects_invalid():
    assert data._coerce_limit_entry(None) is None
    assert data._coerce_limit_entry("none") is None
    assert data._coerce_limit_entry("10") == 10
    with pytest.raises(ValueError):
        data._coerce_limit_entry("not-a-number")
    with pytest.raises(ValueError):
        data._coerce_limit_entry(-1)


def test_coerce_policy_kwarg_validates_type():
    assert data._coerce_policy_kwarg(None) is None
    assert data._coerce_policy_kwarg("drop") == "drop"
    sentinel = {"*": "drop"}
    assert data._coerce_policy_kwarg(sentinel) is sentinel
    with pytest.raises(TypeError):
        data._coerce_policy_kwarg(42)


def test_coerce_limit_kwarg_accepts_mappings_and_strings():
    assert data._coerce_limit_kwarg(None) is None
    limits = {"col": 5}
    assert data._coerce_limit_kwarg(limits) is limits
    assert data._coerce_limit_kwarg(3) == 3
    assert data._coerce_limit_kwarg(3.0) == 3
    assert data._coerce_limit_kwarg("none") is None
    assert data._coerce_limit_kwarg("7") == 7
    with pytest.raises(TypeError):
        data._coerce_limit_kwarg([])
    with pytest.raises(TypeError):
        data._coerce_limit_kwarg("abc")


def test_finalise_validated_frame_includes_metadata(sample_metadata):
    frame = pd.DataFrame(
        {"FundA": [1.0, 2.0], "FundB": [0.5, 0.25]},
        index=pd.DatetimeIndex(
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            ],
            name="Date",
        ),
    )
    validated = ValidatedMarketData(frame=frame, metadata=sample_metadata)

    with_date = data._finalise_validated_frame(validated, include_date_column=True)
    without_date = data._finalise_validated_frame(validated, include_date_column=False)

    assert list(with_date.columns) == ["Date", "FundA", "FundB"]
    assert with_date.attrs["market_data"]["metadata"] is sample_metadata
    assert with_date.attrs["market_data_mode"] == sample_metadata.mode.value
    assert with_date.attrs["market_data_frequency"] == sample_metadata.frequency
    assert without_date.index.name == "Date"
    assert (
        without_date.attrs["market_data_frequency_label"]
        == sample_metadata.frequency_label
    )


def test_normalise_numeric_strings_converts_percentages_and_parentheses():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Percent": ["50%", "25%"],
            "Comma": ["1,200", "2,400"],
            "Paren": ["(12)", "(0)"],
            "Mixed": ["abc", None],
        }
    )
    cleaned = data._normalise_numeric_strings(raw)

    pd.testing.assert_series_equal(
        cleaned["Percent"],
        pd.Series([0.5, 0.25], name="Percent"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        cleaned["Comma"],
        pd.Series([1200.0, 2400.0], name="Comma"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        cleaned["Paren"],
        pd.Series([-12.0, 0.0], name="Paren"),
        check_dtype=False,
    )
    # Columns that cannot be coerced should remain untouched
    pd.testing.assert_series_equal(
        cleaned["Mixed"],
        pd.Series(["abc", None], name="Mixed", dtype=object),
    )


def test_validate_payload_normalises_policy_and_limit(monkeypatch, sample_metadata):
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "FundA": ["1,200", "2,400"],
            "FundB": ["50%", "25%"],
        }
    )
    calls: dict[str, object] = {}

    def fake_validate(payload, *, source, missing_policy, missing_limit):
        calls["payload"] = payload.copy()
        calls["source"] = source
        calls["missing_policy"] = missing_policy
        calls["missing_limit"] = missing_limit
        return ValidatedMarketData(
            frame=payload.set_index("Date"), metadata=sample_metadata
        )

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    class StarMapping(dict):
        def __contains__(self, item: object) -> bool:
            return item == "*" or super().__contains__(item)

    result = data._validate_payload(
        df,
        origin="unit-test",
        errors="log",
        include_date_column=False,
        missing_policy=StarMapping(
            {"FundA": "Both", "FundB": None, 123: "FFILL", "Other": 42}
        ),
        missing_limit={"FundA": "10", "FundB": None, "*": "5", 456: "0"},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "Date"
    assert result.attrs["market_data_missing_policy"] == sample_metadata.missing_policy
    assert list(result.columns) == ["FundA", "FundB"]

    payload = calls["payload"]
    pd.testing.assert_series_equal(
        payload["FundA"],
        pd.Series([1200.0, 2400.0], name="FundA"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        payload["FundB"],
        pd.Series([0.5, 0.25], name="FundB"),
        check_dtype=False,
    )
    assert calls["source"] == "unit-test"
    assert calls["missing_policy"] == {
        "FundA": "ffill",
        "FundB": "drop",
        "*": "drop",
        "123": "ffill",
        "Other": "42",
    }
    assert calls["missing_limit"] == {
        "FundA": 10,
        "FundB": None,
        "*": 5,
        "456": 0,
    }

    calls.clear()
    data._validate_payload(
        df,
        origin="unit-test",
        errors="log",
        include_date_column=False,
        missing_policy={"FundA": "Both", "*": "zeros"},
        missing_limit=None,
    )
    assert calls["missing_policy"]["*"] == "zero"


def test_validate_payload_logs_errors_and_returns_none(monkeypatch, caplog):
    df = pd.DataFrame({"Date": ["2020-01-01"], "FundA": ["bad"]})

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("Could not be parsed: invalid date")

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    caplog.set_level("ERROR")
    result = data._validate_payload(
        df,
        origin="failing.csv",
        errors="log",
        include_date_column=True,
    )

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_payload_logs_non_parse_errors(monkeypatch, caplog):
    df = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1]})

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("General validation failure")

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    caplog.set_level("ERROR")
    result = data._validate_payload(
        df,
        origin="other.csv",
        errors="log",
        include_date_column=True,
    )

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_validate_payload_propagates_errors(monkeypatch):
    df = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1]})

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    with pytest.raises(MarketDataValidationError):
        data._validate_payload(
            df,
            origin="boom",
            errors="raise",
            include_date_column=True,
        )


def test_load_csv_happy_path(tmp_path, monkeypatch, sample_metadata):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text("Date,FundA\n2020-01-01,1\n", encoding="utf-8")

    sentinel = pd.DataFrame({"FundA": [1.0]})

    def fake_validate(payload, **kwargs):
        assert kwargs["origin"] == str(csv_path)
        assert kwargs["include_date_column"] is False
        assert kwargs["missing_policy"] == "Both"
        assert kwargs["missing_limit"] == 2
        return sentinel

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    result = data.load_csv(
        str(csv_path),
        include_date_column=False,
        nan_policy="Both",
        nan_limit="2",
    )

    pd.testing.assert_frame_equal(result, sentinel)


def test_load_csv_returns_none_for_missing_file(caplog):
    caplog.set_level("ERROR")
    result = data.load_csv("/non/existent.csv")
    assert result is None
    assert "/non/existent.csv" in caplog.text


def test_load_csv_logs_permission_issue(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "restricted.csv"
    csv_path.write_text("Date,FundA\n2020-01-01,1\n", encoding="utf-8")

    caplog.set_level("ERROR")
    monkeypatch.setattr(data, "_is_readable", lambda mode: False)

    result = data.load_csv(str(csv_path))

    assert result is None
    assert "Permission denied accessing file" in caplog.text


def test_load_csv_raises_permission_error_when_requested(tmp_path, monkeypatch):
    csv_path = tmp_path / "restricted_raise.csv"
    csv_path.write_text("Date,FundA\n2020-01-01,1\n", encoding="utf-8")

    monkeypatch.setattr(data, "_is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        data.load_csv(str(csv_path), errors="raise")


def test_load_csv_handles_directory_target(tmp_path, caplog):
    directory = tmp_path / "dir"
    directory.mkdir()

    caplog.set_level("ERROR")
    result = data.load_csv(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_load_csv_logs_market_validation_errors(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text("Date,FundA\n2020-01-01,1\n", encoding="utf-8")

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("Could not be parsed: missing date")

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    caplog.set_level("ERROR")
    result = data.load_csv(str(csv_path))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_logs_non_parse_market_errors(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "market2.csv"
    csv_path.write_text("Date,FundA\n2020-01-01,1\n", encoding="utf-8")

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("General validation failure")

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    caplog.set_level("ERROR")
    result = data.load_csv(str(csv_path))

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_load_parquet_propagates_permission_error(tmp_path, monkeypatch):
    parquet_path = tmp_path / "market.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(data, "_is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        data.load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_happy_path(tmp_path, monkeypatch):
    parquet_path = tmp_path / "market.parquet"
    parquet_path.write_bytes(b"")

    sentinel = pd.DataFrame({"FundA": [1.0]})

    def fake_read_parquet(path):
        assert path == str(parquet_path)
        return pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1]})

    def fake_validate(payload, **kwargs):
        assert kwargs["origin"] == str(parquet_path)
        return sentinel

    monkeypatch.setattr(data.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    result = data.load_parquet(str(parquet_path))
    pd.testing.assert_frame_equal(result, sentinel)


def test_load_parquet_applies_legacy_kwargs(tmp_path, monkeypatch):
    parquet_path = tmp_path / "legacy.parquet"
    parquet_path.write_bytes(b"")

    def fake_read_parquet(path):
        return pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1]})

    def fake_validate(payload, **kwargs):
        assert kwargs["missing_policy"] == "legacy"
        assert kwargs["missing_limit"] == 4
        return pd.DataFrame({"FundA": [1]})

    monkeypatch.setattr(data.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    result = data.load_parquet(
        str(parquet_path),
        nan_policy="legacy",
        nan_limit="4",
    )

    pd.testing.assert_frame_equal(result, pd.DataFrame({"FundA": [1]}))


def test_load_parquet_logs_market_validation_errors(tmp_path, monkeypatch, caplog):
    parquet_path = tmp_path / "invalid.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(data.pd, "read_parquet", lambda path: pd.DataFrame())

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("Unable to parse dates")

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    caplog.set_level("ERROR")
    result = data.load_parquet(str(parquet_path))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_logs_non_parse_market_errors(tmp_path, monkeypatch, caplog):
    parquet_path = tmp_path / "invalid2.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(data.pd, "read_parquet", lambda path: pd.DataFrame())

    def fake_validate(*_args, **_kwargs):
        raise MarketDataValidationError("General validation failure")

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    caplog.set_level("ERROR")
    result = data.load_parquet(str(parquet_path))

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_load_parquet_handles_missing_paths(caplog):
    caplog.set_level("ERROR")
    result = data.load_parquet("/nonexistent.parquet")
    assert result is None
    assert "/nonexistent.parquet" in caplog.text


def test_load_parquet_handles_directory_target(tmp_path, caplog):
    directory = tmp_path / "pq_dir"
    directory.mkdir()

    caplog.set_level("ERROR")
    result = data.load_parquet(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_validate_dataframe_delegates(monkeypatch):
    sentinel = pd.DataFrame({"FundA": [1]})

    def fake_validate(payload, **kwargs):
        assert kwargs["origin"] == "dataframe"
        return sentinel

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    df = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1]})
    result = data.validate_dataframe(df, include_date_column=False)
    pd.testing.assert_frame_equal(result, sentinel)


def test_identify_risk_free_fund_picks_lowest_standard_deviation(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "FundA": [1.0, 1.2],
            "FundB": [0.5, 0.6],
            "Text": ["x", "y"],
        }
    )
    assert data.identify_risk_free_fund(df) == "FundB"
    assert "Risk-free column: FundB" in caplog.text


def test_identify_risk_free_fund_returns_none_without_numeric_columns():
    df = pd.DataFrame({"Date": ["2020-01-01"], "Label": ["x"]})
    assert data.identify_risk_free_fund(df) is None


def test_ensure_datetime_parses_known_format():
    df = pd.DataFrame({"Date": ["01/31/25", "02/01/25"]})
    converted = data.ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])


def test_ensure_datetime_falls_back_to_generic_parsing():
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"]})
    converted = data.ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])


def test_ensure_datetime_skips_when_column_missing():
    df = pd.DataFrame({"Other": [1, 2]})
    result = data.ensure_datetime(df.copy())
    assert result.equals(df)


def test_ensure_datetime_noop_for_existing_datetime():
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"])})
    result = data.ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_raises_on_malformed_dates(caplog):
    df = pd.DataFrame({"Date": ["not-a-date", "still bad"]})
    caplog.set_level("ERROR")
    with pytest.raises(ValueError):
        data.ensure_datetime(df)
    assert "malformed date(s)" in caplog.text.lower()


@pytest.mark.parametrize(
    "mode, expected",
    [
        (0o000, False),
        (0o400, True),
        (0o040, True),
        (0o004, True),
    ],
)
def test_is_readable_checks_permission_bits(mode, expected):
    assert data._is_readable(mode) is expected
