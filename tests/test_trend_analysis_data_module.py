from __future__ import annotations

import stat

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trend_analysis import data as data_module
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
)
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    ValidatedMarketData,
)


def _build_metadata(index: pd.DatetimeIndex) -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="daily",
        frequency_detected="D",
        frequency_label="Daily",
        frequency_median_spacing_days=1.0,
        frequency_missing_periods=0,
        frequency_max_gap_periods=0,
        frequency_tolerance_periods=0,
        start=index[0].to_pydatetime(),
        end=index[-1].to_pydatetime(),
        rows=len(index),
        columns=["alpha", "beta"],
        missing_policy=DEFAULT_POLICY_FALLBACK,
        missing_policy_limit=None,
    )


def test_normalise_policy_alias_variants():
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias(" ") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("both") == "ffill"
    assert _normalise_policy_alias("BackFill") == "ffill"
    assert _normalise_policy_alias("zero_fill") == "zero"
    assert _normalise_policy_alias("custom") == "custom"


def test_coerce_limit_entry_validates_bounds():
    assert _coerce_limit_entry(None) is None
    assert _coerce_limit_entry("none") is None
    assert _coerce_limit_entry("10") == 10
    with pytest.raises(ValueError):
        _coerce_limit_entry("invalid")
    with pytest.raises(ValueError):
        _coerce_limit_entry(-2)


def test_coerce_policy_and_limit_kwargs():
    mapping = {"x": "ffill"}
    assert _coerce_policy_kwarg(None) is None
    assert _coerce_policy_kwarg("drop") == "drop"
    assert _coerce_policy_kwarg(mapping) is mapping
    with pytest.raises(TypeError):
        _coerce_policy_kwarg(["drop"])  # type: ignore[arg-type]

    assert _coerce_limit_kwarg(None) is None
    assert _coerce_limit_kwarg(5) == 5
    assert _coerce_limit_kwarg(5.0) == 5
    assert _coerce_limit_kwarg("none") is None
    overrides = {"a": 1, "b": None}
    assert _coerce_limit_kwarg(overrides) is overrides
    with pytest.raises(TypeError):
        _coerce_limit_kwarg([1, 2])  # type: ignore[arg-type]


def test_coerce_limit_kwarg_rejects_non_numeric_string():
    with pytest.raises(TypeError):
        _coerce_limit_kwarg("not-a-number")


def test_normalise_numeric_strings_handles_percent_and_signs():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "alpha": ["1,000", "(2,500)"],
            "beta": ["25%", "-50%"],
            "gamma": [1.0, 2.0],
        }
    )

    cleaned = _normalise_numeric_strings(frame)

    assert list(cleaned["alpha"]) == [1000.0, -2500.0]
    assert list(cleaned["beta"]) == [0.25, -0.5]
    assert list(cleaned["gamma"]) == [1.0, 2.0]


def test_validate_payload_success(monkeypatch):
    source = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "alpha": ["1,000", "2,000"],
            "beta": ["25%", "(50%)"],
        }
    )

    index = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
    validated_frame = pd.DataFrame(
        {"alpha": [1000.0, 2000.0], "beta": [0.25, -0.5]}, index=index
    )
    validated_frame.index.name = "Date"
    metadata = _build_metadata(index)
    validated = ValidatedMarketData(validated_frame, metadata)

    captured: dict[str, object] = {}

    def fake_validate(payload, *, missing_policy, missing_limit, source: str, **kwargs):
        captured["payload"] = payload
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        captured["source"] = source
        return validated

    monkeypatch.setattr(data_module, "validate_market_data", fake_validate)

    result = _validate_payload(
        source,
        origin="fixture.csv",
        errors="log",
        include_date_column=False,
        missing_policy={"alpha": "both", "*": None},
        missing_limit={"alpha": "10", "beta": "none"},
    )

    payload = captured["payload"]
    assert list(payload.columns) == ["Date", "alpha", "beta"]
    expected_dates = pd.Series(index.tz_localize(None), name="Date")
    pd.testing.assert_series_equal(
        pd.to_datetime(payload["Date"]),
        expected_dates,
        check_names=False,
    )
    assert list(payload["alpha"]) == [1000.0, 2000.0]
    assert list(payload["beta"]) == [0.25, -0.5]
    assert captured["missing_policy"] == {
        "alpha": "ffill",
        "*": DEFAULT_POLICY_FALLBACK,
    }
    assert captured["missing_limit"] == {"alpha": 10, "beta": None}
    assert captured["source"] == "fixture.csv"

    assert_frame_equal(result, validated_frame)
    market_attrs = result.attrs["market_data"]
    assert market_attrs["metadata"] == metadata
    assert result.attrs["market_data_mode"] == metadata.mode.value
    assert result.attrs["market_data_missing_policy"] == metadata.missing_policy
    assert result.attrs["market_data_columns"] == metadata.columns


def test_validate_payload_logs_and_suppresses_error(monkeypatch, caplog):
    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [1]})

    def fake_validate(*args, **kwargs):
        raise MarketDataValidationError("Unable to parse something")

    monkeypatch.setattr(data_module, "validate_market_data", fake_validate)

    with caplog.at_level("ERROR"):
        result = _validate_payload(
            df, origin="bad.csv", errors="log", include_date_column=True
        )

    assert result is None
    assert "Unable to parse Date values in bad.csv" in caplog.text


def test_validate_payload_reraises_when_requested(monkeypatch):
    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [1]})

    monkeypatch.setattr(
        data_module,
        "validate_market_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MarketDataValidationError("fail")
        ),
    )

    with pytest.raises(MarketDataValidationError):
        _validate_payload(
            df, origin="bad.csv", errors="raise", include_date_column=True
        )


def test_finalise_validated_frame_includes_date_column():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
    frame = pd.DataFrame({"alpha": [1.0, 2.0]}, index=index)
    frame.index.name = "Date"
    metadata = _build_metadata(index)
    validated = ValidatedMarketData(frame, metadata)

    result = _finalise_validated_frame(validated, include_date_column=True)

    assert "Date" in result.columns
    assert result.attrs["market_data_rows"] == metadata.rows


def test_is_readable_checks_permission_bits():
    readable_mode = stat.S_IRUSR | stat.S_IFREG
    assert _is_readable(readable_mode) is True
    assert _is_readable(0) is False


def test_load_csv_validates_payload(tmp_path, monkeypatch):
    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [10]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    index = pd.to_datetime(["2024-01-01"], utc=True)
    validated = ValidatedMarketData(
        pd.DataFrame({"alpha": [10]}, index=index, columns=["alpha"]),
        _build_metadata(index),
    )
    validated.frame.index.name = "Date"

    captured: dict[str, object] = {}

    def fake_validate(payload, **kwargs):
        captured["kwargs"] = kwargs
        return validated

    monkeypatch.setattr(data_module, "validate_market_data", fake_validate)

    result = data_module.load_csv(
        str(path),
        nan_policy="zeros",
        nan_limit="3",
        missing_limit=None,
    )

    assert "Date" in result.columns
    args = captured["kwargs"]
    assert args["missing_policy"] == "zero"
    assert args["missing_limit"] == 3


def test_load_csv_handles_permission_error(tmp_path, monkeypatch, caplog):
    path = tmp_path / "restricted.csv"
    path.write_text("alpha\n1")

    monkeypatch.setattr(data_module, "_is_readable", lambda mode: False)

    with caplog.at_level("ERROR"):
        result = data_module.load_csv(str(path), errors="log")

    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_missing_file_returns_none(caplog):
    with caplog.at_level("ERROR"):
        result = data_module.load_csv("nonexistent.csv", errors="log")

    assert result is None


def test_load_parquet_success(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    path.write_text("placeholder")

    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [1]})
    monkeypatch.setattr(data_module, "_is_readable", lambda mode: True)
    monkeypatch.setattr(pd, "read_parquet", lambda p: df)

    index = pd.to_datetime(["2024-01-01"], utc=True)
    validated = ValidatedMarketData(
        pd.DataFrame({"alpha": [1]}, index=index, columns=["alpha"]),
        _build_metadata(index),
    )
    validated.frame.index.name = "Date"
    monkeypatch.setattr(
        data_module, "validate_market_data", lambda payload, **kwargs: validated
    )

    result = data_module.load_parquet(str(path))

    assert "Date" in result.columns


def test_load_csv_permission_raise(tmp_path, monkeypatch):
    path = tmp_path / "blocked.csv"
    path.write_text("alpha\n1")
    monkeypatch.setattr(data_module, "_is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        data_module.load_csv(str(path), errors="raise")


def test_load_csv_directory_logs(tmp_path, caplog):
    directory = tmp_path / "folder"
    directory.mkdir()

    with caplog.at_level("ERROR"):
        result = data_module.load_csv(str(directory), errors="log")

    assert result is None


def test_load_csv_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        data_module.load_csv(str(missing), errors="raise")


def test_load_csv_market_data_validation_error(monkeypatch, tmp_path, caplog):
    path = tmp_path / "data.csv"
    path.write_text("alpha\n1")

    def raise_validation(*args, **kwargs):
        raise MarketDataValidationError("Could not be parsed header")

    monkeypatch.setattr(data_module, "_validate_payload", raise_validation)

    with caplog.at_level("ERROR"):
        result = data_module.load_csv(str(path), errors="log")

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_legacy_missing_limit(tmp_path, monkeypatch):
    path = tmp_path / "legacy.csv"
    path.write_text("alpha\n1")

    index = pd.to_datetime(["2024-01-01"], utc=True)
    validated = ValidatedMarketData(
        pd.DataFrame({"alpha": [1]}, index=index, columns=["alpha"]),
        _build_metadata(index),
    )
    validated.frame.index.name = "Date"

    monkeypatch.setattr(
        data_module, "validate_market_data", lambda payload, **kwargs: validated
    )

    data_module.load_csv(
        str(path),
        missing_policy=None,
        missing_limit=5,
        nan_limit="2",
        nan_policy="drop",
    )


def test_load_parquet_permission_error_raises(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    path.write_text("placeholder")

    monkeypatch.setattr(data_module, "_is_readable", lambda mode: False)

    with pytest.raises(PermissionError):
        data_module.load_parquet(str(path), errors="raise")


def test_load_parquet_missing_file_logging(caplog):
    with caplog.at_level("ERROR"):
        result = data_module.load_parquet("absent.parquet", errors="log")

    assert result is None
    assert "absent.parquet" in caplog.text


def test_load_parquet_missing_file_raises(tmp_path):
    missing = tmp_path / "absent.parquet"
    with pytest.raises(FileNotFoundError):
        data_module.load_parquet(str(missing), errors="raise")


def test_load_parquet_market_data_validation_error(monkeypatch, tmp_path, caplog):
    path = tmp_path / "validate.parquet"
    path.write_text("placeholder")

    monkeypatch.setattr(data_module, "_is_readable", lambda mode: True)
    monkeypatch.setattr(pd, "read_parquet", lambda p: pd.DataFrame())
    monkeypatch.setattr(
        data_module,
        "_validate_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MarketDataValidationError("Could not be parsed rows")
        ),
    )

    with caplog.at_level("ERROR"):
        result = data_module.load_parquet(str(path), errors="log")

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_legacy_kwargs(tmp_path, monkeypatch):
    path = tmp_path / "legacy.parquet"
    path.write_text("placeholder")

    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [1]})
    monkeypatch.setattr(data_module, "_is_readable", lambda mode: True)
    monkeypatch.setattr(pd, "read_parquet", lambda p: df)

    index = pd.to_datetime(["2024-01-01"], utc=True)
    validated = ValidatedMarketData(
        pd.DataFrame({"alpha": [1]}, index=index, columns=["alpha"]),
        _build_metadata(index),
    )
    validated.frame.index.name = "Date"
    monkeypatch.setattr(
        data_module, "validate_market_data", lambda payload, **kwargs: validated
    )

    data_module.load_parquet(
        str(path),
        nan_policy="drop",
        nan_limit="4",
        missing_limit=None,
    )


def test_validate_dataframe_delegates(monkeypatch):
    df = pd.DataFrame({"Date": ["2024-01-01"], "alpha": [1]})

    called = {}

    def fake_validate(payload, **kwargs):
        called.update(kwargs)
        idx = pd.to_datetime(["2024-01-01"], utc=True)
        idx.name = "Date"
        return ValidatedMarketData(
            pd.DataFrame({"alpha": [1]}, index=idx),
            _build_metadata(idx),
        )

    monkeypatch.setattr(data_module, "validate_market_data", fake_validate)

    data_module.validate_dataframe(df, origin="manual", errors="raise")

    assert called["source"] == "manual"


def test_identify_risk_free_fund_selects_lowest_std():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "alpha": [1, 1, 1, 1],
            "beta": [0, 2, 4, 6],
        }
    )

    assert data_module.identify_risk_free_fund(df) == "alpha"


def test_identify_risk_free_fund_handles_no_numeric():
    df = pd.DataFrame({"Date": ["2024-01-01"], "text": ["value"]})
    assert data_module.identify_risk_free_fund(df) is None


def test_ensure_datetime_parses_strings(caplog):
    df = pd.DataFrame({"Date": ["01/31/24", "02/29/24"], "alpha": [1, 2]})

    result = data_module.ensure_datetime(df)

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_generic_parse_success():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "alpha": [1, 2]})

    result = data_module.ensure_datetime(df)

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_raises_on_malformed(caplog):
    df = pd.DataFrame({"Date": ["bad-date"], "alpha": [1]})

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            data_module.ensure_datetime(df)

    assert "malformed date" in caplog.text.lower()


def test_ensure_datetime_noop_for_datetime_column():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"Date": dates, "alpha": [1, 2]})

    result = data_module.ensure_datetime(df)

    assert result is df
