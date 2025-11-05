import logging
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Mapping
from unittest.mock import MagicMock

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
        frequency_label="daily",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        rows=3,
        columns=["AAA", "BBB"],
        missing_policy="drop",
        missing_policy_limit=5,
        missing_policy_summary="drop:AAA",
    )


@pytest.fixture()
def validated_payload(sample_metadata: MarketDataMetadata) -> ValidatedMarketData:
    frame = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [2.0, 4.0, 6.0],
        },
        index=pd.DatetimeIndex(
            [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            name="Date",
        ),
    )
    frame.attrs["custom"] = "kept"
    return ValidatedMarketData(frame=frame, metadata=sample_metadata)


def test_normalise_policy_alias_variants():
    assert data._normalise_policy_alias(None) == data.DEFAULT_POLICY_FALLBACK
    assert data._normalise_policy_alias("  ") == data.DEFAULT_POLICY_FALLBACK
    assert data._normalise_policy_alias("Both") == "ffill"
    assert data._normalise_policy_alias("zero_fill") == "zero"
    assert data._normalise_policy_alias("DROP") == "drop"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("none", None),
        (10, 10),
        ("7", 7),
    ],
)
def test_coerce_limit_entry_valid(value, expected):
    assert data._coerce_limit_entry(value) == expected


@pytest.mark.parametrize("value", ["abc", {}])
def test_coerce_limit_entry_invalid(value):
    with pytest.raises(ValueError):
        data._coerce_limit_entry(value)


def test_coerce_limit_entry_negative():
    with pytest.raises(ValueError):
        data._coerce_limit_entry(-1)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("policy", "policy"),
        ({"AAA": "drop"}, {"AAA": "drop"}),
    ],
)
def test_coerce_policy_kwarg_valid(value, expected):
    assert data._coerce_policy_kwarg(value) == expected


@pytest.mark.parametrize("value", [123, ["drop"]])
def test_coerce_policy_kwarg_invalid(value):
    with pytest.raises(TypeError):
        data._coerce_policy_kwarg(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (5, 5),
        (3.0, 3),
        ("9", 9),
        ("none", None),
        ({"AAA": 1, "BBB": "none"}, {"AAA": 1, "BBB": "none"}),
    ],
)
def test_coerce_limit_kwarg_valid(value, expected):
    assert data._coerce_limit_kwarg(value) == expected


@pytest.mark.parametrize("value", [object(), "not-int", [1, 2, 3]])
def test_coerce_limit_kwarg_invalid(value):
    with pytest.raises(TypeError):
        data._coerce_limit_kwarg(value)


@pytest.mark.parametrize(
    "mode, readable",
    [
        (stat.S_IRUSR, True),
        (stat.S_IRGRP, True),
        (stat.S_IROTH, True),
        (0, False),
    ],
)
def test_is_readable(mode, readable):
    assert data._is_readable(mode) is readable


def test_finalise_validated_frame_includes_metadata(validated_payload):
    result = data._finalise_validated_frame(validated_payload, include_date_column=True)
    assert list(result.columns) == ["Date", "AAA", "BBB"]
    assert result.attrs["custom"] == "kept"
    market_attrs = result.attrs["market_data"]
    assert market_attrs["metadata"] is validated_payload.metadata
    assert result.attrs["market_data_mode"] == validated_payload.metadata.mode.value
    assert result.attrs["market_data_missing_policy_summary"] == "drop:AAA"


def test_finalise_validated_frame_without_date(validated_payload):
    result = data._finalise_validated_frame(
        validated_payload, include_date_column=False
    )
    assert "Date" not in result.columns
    assert result.index.equals(validated_payload.frame.index)


def test_normalise_numeric_strings_handles_formats():
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "plain": ["1", "2"],
            "commas": ["1,234", "2,345"],
            "percent": ["10%", "20%"],
            "paren": ["(5)", "(10)"],
            "mixed": ["abc", "3"],
            "already_numeric": [1.0, 2.0],
            "all_text": ["abc", "def"],
        }
    )

    cleaned = data._normalise_numeric_strings(frame)

    assert cleaned["plain"].tolist() == [1.0, 2.0]
    assert cleaned["commas"].tolist() == [1234.0, 2345.0]
    assert cleaned["percent"].tolist() == [0.1, 0.2]
    assert cleaned["paren"].tolist() == [-5.0, -10.0]
    assert cleaned["mixed"].isna().iloc[0]
    assert cleaned["mixed"].iloc[1] == 3.0
    assert cleaned["already_numeric"].tolist() == [1.0, 2.0]
    assert cleaned["all_text"].tolist() == ["abc", "def"]


class _WeirdPolicy(dict):
    """Mapping that reports a ``*`` key without yielding it in ``items()``."""

    def __contains__(self, key):  # pragma: no cover - behaviour verified via use
        if key == "*":
            return True
        return super().__contains__(key)


def test_validate_payload_policy_coercions(monkeypatch, validated_payload):
    payload = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["3"]})

    policy = _WeirdPolicy({"AAA": None, "BBB": 5})
    limit = {"AAA": "10", "BBB": None}

    def fake_validate(frame: pd.DataFrame, *, missing_policy, missing_limit, **kwargs):
        assert missing_policy["AAA"] == data.DEFAULT_POLICY_FALLBACK
        assert missing_policy["BBB"] == "5"
        assert missing_policy["*"] == data.DEFAULT_POLICY_FALLBACK
        assert missing_limit == {"AAA": 10, "BBB": None}
        return validated_payload

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    result = data._validate_payload(
        payload,
        origin="policy.csv",
        errors="log",
        include_date_column=True,
        missing_policy=policy,
        missing_limit=limit,
    )

    assert result.attrs["market_data_columns"] == list(
        validated_payload.metadata.columns
    )


def test_validate_payload_success(monkeypatch, validated_payload):
    payload = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "AAA": ["1", "2"],
        }
    )

    def fake_validate(
        frame: pd.DataFrame,
        *,
        source: str,
        missing_policy: Mapping[str, str] | str,
        missing_limit: Mapping[str, int | None] | int | None,
    ) -> ValidatedMarketData:
        assert source == "origin.csv"
        assert isinstance(missing_policy, dict)
        assert missing_policy["AAA"] == "ffill"
        assert missing_policy["*"] == data.DEFAULT_POLICY_FALLBACK
        assert isinstance(missing_limit, dict)
        assert missing_limit["AAA"] is None
        assert missing_limit["*"] == 5
        # Ensure numeric strings normalised before validation
        assert frame["AAA"].tolist() == [1.0, 2.0]
        return validated_payload

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    result = data._validate_payload(
        payload,
        origin="origin.csv",
        errors="log",
        include_date_column=True,
        missing_policy={"AAA": "bfill", "*": "drop"},
        missing_limit={"AAA": "none", "*": "5"},
    )

    assert list(result.columns) == ["Date", "AAA", "BBB"]
    assert result.attrs["market_data_rows"] == validated_payload.metadata.rows


def test_validate_payload_missing_policy_string(monkeypatch, validated_payload):
    payload = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["3"]})

    def fake_validate(frame: pd.DataFrame, *, source: str, **kwargs):
        assert kwargs["missing_policy"] == "ffill"
        assert kwargs["missing_limit"] == 2
        return validated_payload

    monkeypatch.setattr(data, "validate_market_data", fake_validate)

    result = data._validate_payload(
        payload,
        origin="inline",
        errors="log",
        include_date_column=False,
        missing_policy="both",
        missing_limit="2",
    )

    assert result.index.equals(validated_payload.frame.index)


def test_validate_payload_logs_market_data_error(monkeypatch, caplog):
    payload = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["3"]})

    monkeypatch.setattr(
        data,
        "validate_market_data",
        MagicMock(
            side_effect=MarketDataValidationError("Date column could not be parsed")
        ),
    )

    with caplog.at_level(logging.ERROR):
        result = data._validate_payload(
            payload,
            origin="broken.csv",
            errors="log",
            include_date_column=True,
        )

    assert result is None
    assert "Unable to parse Date values in broken.csv" in caplog.text


def test_validate_payload_logs_without_parse_hint(monkeypatch, caplog):
    payload = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["3"]})

    monkeypatch.setattr(
        data,
        "validate_market_data",
        MagicMock(side_effect=MarketDataValidationError("Validation failed")),
    )

    with caplog.at_level(logging.ERROR):
        result = data._validate_payload(
            payload,
            origin="broken.csv",
            errors="log",
            include_date_column=True,
        )

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_validate_payload_raises_when_requested(monkeypatch):
    payload = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["3"]})
    monkeypatch.setattr(
        data,
        "validate_market_data",
        MagicMock(side_effect=MarketDataValidationError("bad")),
    )

    with pytest.raises(MarketDataValidationError):
        data._validate_payload(
            payload,
            origin="broken.csv",
            errors="raise",
            include_date_column=True,
        )


def test_load_csv_success(tmp_path, monkeypatch):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    sentinel = object()
    called_kwargs = {}

    def fake_validate(raw: pd.DataFrame, **kwargs):
        called_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    result = data.load_csv(str(csv_path), missing_policy="drop")

    assert result is sentinel
    assert called_kwargs["missing_policy"] == "drop"
    assert called_kwargs["origin"] == str(csv_path)


def test_load_csv_legacy_kwargs(tmp_path, monkeypatch):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "legacy.csv"
    frame.to_csv(csv_path, index=False)

    captured = {}

    def fake_validate(raw: pd.DataFrame, **kwargs):
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(data, "_validate_payload", fake_validate)

    data.load_csv(
        str(csv_path),
        nan_policy="zeros",
        nan_limit="3",
        missing_limit="4",
    )

    assert captured["missing_policy"] == "zeros"
    assert captured["missing_limit"] == "4"


def test_load_csv_legacy_nan_limit(tmp_path, monkeypatch):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "legacy_fallback.csv"
    frame.to_csv(csv_path, index=False)

    captured = {}
    monkeypatch.setattr(
        data, "_validate_payload", lambda raw, **kwargs: captured.update(kwargs) or raw
    )

    data.load_csv(str(csv_path), nan_limit="7")

    assert captured["missing_limit"] == 7


def test_load_csv_missing_file_logs_error(monkeypatch, caplog):
    missing_path = "nonexistent.csv"
    monkeypatch.setattr(Path, "exists", MagicMock(return_value=False))

    with caplog.at_level(logging.ERROR):
        result = data.load_csv(missing_path)

    assert result is None
    assert missing_path in caplog.text


def test_load_csv_permission_denied(monkeypatch, tmp_path, caplog):
    csv_path = tmp_path / "restricted.csv"
    csv_path.write_text("Date,AAA\n2024-01-01,1\n")

    monkeypatch.setattr(data, "_validate_payload", MagicMock(return_value=None))
    monkeypatch.setattr(data, "_is_readable", MagicMock(return_value=False))

    with caplog.at_level(logging.ERROR):
        result = data.load_csv(str(csv_path))

    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_permission_raise(monkeypatch, tmp_path):
    csv_path = tmp_path / "restricted.csv"
    csv_path.write_text("Date,AAA\n2024-01-01,1\n")

    monkeypatch.setattr(data, "_is_readable", MagicMock(return_value=False))

    with pytest.raises(PermissionError):
        data.load_csv(str(csv_path), errors="raise")


def test_load_csv_raises_when_requested(monkeypatch):
    missing_path = "missing.csv"
    monkeypatch.setattr(Path, "exists", MagicMock(return_value=False))

    with pytest.raises(FileNotFoundError):
        data.load_csv(missing_path, errors="raise")


def test_load_csv_directory_error(monkeypatch, tmp_path, caplog):
    directory = tmp_path / "folder"
    directory.mkdir()

    with caplog.at_level(logging.ERROR):
        result = data.load_csv(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_load_csv_handles_validation_error(monkeypatch, tmp_path, caplog):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "bad.csv"
    frame.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("Unable to parse Date")),
    )

    with caplog.at_level(logging.ERROR):
        result = data.load_csv(str(csv_path))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_handles_validation_error_without_hint(monkeypatch, tmp_path, caplog):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "bad_plain.csv"
    frame.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("Other failure")),
    )

    with caplog.at_level(logging.ERROR):
        result = data.load_csv(str(csv_path))

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_load_csv_handles_validation_error_raise(monkeypatch, tmp_path):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    csv_path = tmp_path / "bad_raise.csv"
    frame.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("cannot parse")),
    )

    with pytest.raises(MarketDataValidationError):
        data.load_csv(str(csv_path), errors="raise")


def test_load_parquet_success(tmp_path, monkeypatch):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")

    raw = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [2]})
    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=raw))

    sentinel = object()
    monkeypatch.setattr(data, "_validate_payload", MagicMock(return_value=sentinel))

    result = data.load_parquet(str(parquet_path), missing_limit=3)

    assert result is sentinel


def test_load_parquet_legacy_kwargs(tmp_path, monkeypatch):
    parquet_path = tmp_path / "legacy.parquet"
    parquet_path.write_bytes(b"")

    raw = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [2]})
    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=raw))

    captured = {}
    monkeypatch.setattr(
        data, "_validate_payload", lambda *_args, **kwargs: captured.update(kwargs)
    )

    data.load_parquet(
        str(parquet_path),
        nan_policy="bfill",
        nan_limit="8",
        missing_limit="9",
    )

    assert captured["missing_policy"] == "bfill"
    assert captured["missing_limit"] == "9"


def test_load_parquet_legacy_nan_limit(tmp_path, monkeypatch):
    parquet_path = tmp_path / "legacy_fallback.parquet"
    parquet_path.write_bytes(b"")

    raw = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [2]})
    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=raw))

    captured = {}
    monkeypatch.setattr(
        data, "_validate_payload", lambda *_args, **kwargs: captured.update(kwargs)
    )

    data.load_parquet(str(parquet_path), nan_limit="6")

    assert captured["missing_limit"] == 6


def test_load_parquet_permission_error(monkeypatch, tmp_path):
    parquet_path = tmp_path / "unreadable.parquet"
    parquet_path.write_bytes(b"")

    mode = os.stat(parquet_path).st_mode
    monkeypatch.setattr(
        Path, "stat", MagicMock(return_value=os.stat_result((mode,) + (0,) * 9))
    )
    monkeypatch.setattr(data, "_is_readable", MagicMock(return_value=False))

    with pytest.raises(PermissionError):
        data.load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_handles_validation_error(monkeypatch, tmp_path, caplog):
    parquet_path = tmp_path / "bad.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("Could not be parsed")),
    )

    with caplog.at_level(logging.ERROR):
        result = data.load_parquet(str(parquet_path))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_handles_validation_error_without_hint(
    monkeypatch, tmp_path, caplog
):
    parquet_path = tmp_path / "bad_plain.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("Other failure")),
    )

    with caplog.at_level(logging.ERROR):
        result = data.load_parquet(str(parquet_path))

    assert result is None
    assert "Unable to parse" not in caplog.text


def test_load_parquet_handles_validation_error_raise(monkeypatch, tmp_path):
    parquet_path = tmp_path / "bad2.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(
        data,
        "_validate_payload",
        MagicMock(side_effect=MarketDataValidationError("bad")),
    )

    with pytest.raises(MarketDataValidationError):
        data.load_parquet(str(parquet_path), errors="raise")


def test_load_parquet_missing_file_logs(monkeypatch, caplog):
    monkeypatch.setattr(Path, "exists", MagicMock(return_value=False))

    with caplog.at_level(logging.ERROR):
        result = data.load_parquet("missing.parquet")

    assert result is None
    assert "missing.parquet" in caplog.text


def test_load_parquet_directory_error(monkeypatch, tmp_path, caplog):
    directory = tmp_path / "dir"
    directory.mkdir()

    with caplog.at_level(logging.ERROR):
        result = data.load_parquet(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_load_parquet_empty_data_error(monkeypatch, tmp_path, caplog):
    parquet_path = tmp_path / "empty.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(
        pd, "read_parquet", MagicMock(side_effect=pd.errors.EmptyDataError("empty"))
    )

    with caplog.at_level(logging.ERROR):
        result = data.load_parquet(str(parquet_path))

    assert result is None
    assert "empty" in caplog.text


def test_validate_dataframe_delegates(monkeypatch):
    frame = pd.DataFrame({"Date": ["2024-01-01"], "AAA": [1]})
    sentinel = object()
    monkeypatch.setattr(data, "_validate_payload", MagicMock(return_value=sentinel))

    result = data.validate_dataframe(frame, include_date_column=False)

    assert result is sentinel


def test_identify_risk_free_fund_returns_none_for_non_numeric():
    df = pd.DataFrame({"Date": ["2024-01-01"], "AAA": ["x"]})
    assert data.identify_risk_free_fund(df) is None


def test_identify_risk_free_fund_selects_lowest_std(caplog):
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAA": [1.0, 1.5, 1.3],
            "BBB": [2.0, 3.0, 5.0],
        }
    )

    with caplog.at_level(logging.INFO):
        result = data.identify_risk_free_fund(df)

    assert result == "AAA"
    assert "Risk-free column: AAA" in caplog.text


def test_ensure_datetime_parses_known_format():
    df = pd.DataFrame({"Date": ["01/01/24", "01/02/24"]})
    converted = data.ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])


def test_ensure_datetime_raises_on_malformed_dates(caplog):
    df = pd.DataFrame({"Date": ["01-01-2024", "bad"]})

    with pytest.raises(ValueError):
        data.ensure_datetime(df)

    assert "malformed" in caplog.text.lower()


def test_ensure_datetime_generic_parse_without_errors():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"]})
    converted = data.ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])


def test_ensure_datetime_noop_when_column_missing():
    df = pd.DataFrame({"Other": [1, 2]})
    result = data.ensure_datetime(df.copy())
    assert result.equals(df)
