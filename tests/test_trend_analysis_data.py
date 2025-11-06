from __future__ import annotations

import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis import data
from trend_analysis.data import (
    DEFAULT_POLICY_FALLBACK,
    _coerce_limit_entry,
    _coerce_limit_kwarg,
    _coerce_policy_kwarg,
    _finalise_validated_frame,
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


def _metadata(columns: list[str], rows: int = 2) -> MarketDataMetadata:
    return MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="D",
        frequency_label="daily",
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 2),
        rows=rows,
        columns=list(columns),
        missing_policy="drop",
        missing_policy_limit=5,
        missing_policy_summary="all good",
    )


class ValidationProbe:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        payload: pd.DataFrame,
        *,
        source: str,
        missing_policy: str | dict[str, str] | None,
        missing_limit: int | dict[str, int | None] | None,
    ) -> ValidatedMarketData:
        frame = payload.copy()
        if "Date" in frame.columns:
            frame = frame.set_index("Date")
        metadata = _metadata(list(frame.columns), rows=len(frame))
        self.calls.append(
            {
                "source": source,
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
                "payload": payload.copy(),
            }
        )
        return ValidatedMarketData(frame=frame, metadata=metadata)


def test_normalise_policy_alias_handles_aliases() -> None:
    assert _normalise_policy_alias(None) == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias("") == DEFAULT_POLICY_FALLBACK
    assert _normalise_policy_alias(" backfill ") == "ffill"
    assert _normalise_policy_alias("zeros") == "zero"
    assert _normalise_policy_alias("drop") == "drop"


def test_coerce_limit_entry_accepts_strings_and_validates() -> None:
    assert _coerce_limit_entry(None) is None
    assert _coerce_limit_entry("none") is None
    assert _coerce_limit_entry("7") == 7
    with pytest.raises(ValueError):
        _coerce_limit_entry("not-an-int")
    with pytest.raises(ValueError):
        _coerce_limit_entry(-1)


def test_coerce_policy_kwarg_rejects_invalid_types() -> None:
    sentinel = {"A": "drop"}
    assert _coerce_policy_kwarg(None) is None
    assert _coerce_policy_kwarg("drop") == "drop"
    assert _coerce_policy_kwarg(sentinel) is sentinel
    with pytest.raises(TypeError):
        _coerce_policy_kwarg(42)


def test_coerce_limit_kwarg_handles_numeric_variants() -> None:
    sentinel_map = {"A": 3}
    assert _coerce_limit_kwarg(None) is None
    assert _coerce_limit_kwarg(5) == 5
    assert _coerce_limit_kwarg(5.0) == 5
    assert _coerce_limit_kwarg("7") == 7
    assert _coerce_limit_kwarg(" none ") is None
    assert _coerce_limit_kwarg(sentinel_map) is sentinel_map
    with pytest.raises(TypeError):
        _coerce_limit_kwarg(object())


def test_coerce_limit_kwarg_rejects_non_numeric_strings() -> None:
    with pytest.raises(TypeError):
        _coerce_limit_kwarg("not-a-number")


def test_finalise_validated_frame_populates_metadata_attrs() -> None:
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "A": [1.0, 2.0]}
    ).set_index("Date")
    metadata = _metadata(["A"], rows=2)
    validated = ValidatedMarketData(frame=frame, metadata=metadata)

    result = _finalise_validated_frame(validated, include_date_column=True)
    assert list(result.columns) == ["Date", "A"]
    assert result.attrs["market_data"]["metadata"] == metadata
    assert result.attrs["market_data_mode"] == metadata.mode.value
    assert result.attrs["market_data_frequency_label"] == metadata.frequency_label
    assert (
        result.attrs["market_data_missing_policy_limit"]
        == metadata.missing_policy_limit
    )


def test_finalise_validated_frame_without_date_column() -> None:
    index = pd.to_datetime(["2020-02-01", "2020-02-02"])
    frame = pd.DataFrame({"A": [3.0, 4.0]}, index=index)
    frame.index.name = "Date"
    metadata = MarketDataMetadata(
        mode=MarketDataMode.PRICE,
        frequency="W",
        frequency_detected="W",
        frequency_label="weekly",
        frequency_median_spacing_days=5.0,
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
        frequency_tolerance_periods=4,
        start=datetime(2020, 2, 1),
        end=datetime(2020, 2, 2),
        rows=2,
        columns=["A"],
        missing_policy="ffill",
        missing_policy_limit=None,
        missing_policy_summary="filled",
    )
    validated = ValidatedMarketData(frame=frame, metadata=metadata)

    result = _finalise_validated_frame(validated, include_date_column=False)
    assert list(result.columns) == ["A"]
    attrs = result.attrs
    assert attrs["market_data_frequency_code"] == "W"
    assert attrs["market_data_frequency_missing_periods"] == 2
    assert attrs["market_data_frequency_tolerance_periods"] == 4
    assert attrs["market_data_missing_policy_summary"] == "filled"


def test_validate_payload_normalises_strings_and_applies_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)

    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Rate": ["12%", "(3.5%)"],
            "Value": [" 1,200 ", "800"],
        }
    )

    result = _validate_payload(
        raw,
        origin="input.csv",
        errors="log",
        include_date_column=False,
        missing_policy={"Rate": "BackFill", "*": None},
        missing_limit={"Rate": "10", "*": "none"},
    )

    assert result is not None
    assert pytest.approx(result["Rate"].iloc[0], rel=1e-6) == 0.12
    assert pytest.approx(result["Rate"].iloc[1], rel=1e-6) == -0.035
    assert pytest.approx(result["Value"].iloc[0], rel=1e-6) == 1200.0
    assert probe.calls
    call = probe.calls[-1]
    assert call["source"] == "input.csv"
    assert call["missing_policy"] == {"Rate": "ffill", "*": DEFAULT_POLICY_FALLBACK}
    assert call["missing_limit"] == {"Rate": 10, "*": None}


def test_validate_payload_accepts_scalar_policy_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)
    raw = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Value": [1, 2]})

    result = _validate_payload(
        raw,
        origin="payload",
        errors="log",
        include_date_column=True,
        missing_policy=" zeros ",
        missing_limit="5",
    )

    assert result is not None
    call = probe.calls[-1]
    assert call["missing_policy"] == "zero"
    assert call["missing_limit"] == 5


def test_validate_payload_supports_non_string_policy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WildcardMapping(dict[str, object]):
        def __contains__(
            self, key: object
        ) -> bool:  # pragma: no cover - exercised via call
            if key == "*":
                return True
            return super().__contains__(key)

    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)
    raw = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Value": [1, 2]})
    policy = WildcardMapping({1: 0})
    limits = {"Value": 2.0}

    result = _validate_payload(
        raw,
        origin="payload",
        errors="log",
        include_date_column=True,
        missing_policy=policy,
        missing_limit=limits,
    )

    assert result is not None
    call = probe.calls[-1]
    assert call["missing_policy"]["1"] == "0"
    assert call["missing_policy"]["*"] == DEFAULT_POLICY_FALLBACK
    assert call["missing_limit"]["Value"] == 2


def test_validate_payload_logs_and_swallows_validation_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def raiser(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Dates could not be parsed", issues=["bad"])

    caplog.set_level("ERROR", "trend_analysis.data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "validate_market_data", raiser)
        frame = pd.DataFrame({"Date": ["bad"], "Value": [1]})
        result = _validate_payload(
            frame,
            origin="payload",
            errors="log",
            include_date_column=True,
        )
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_payload_logs_generic_validation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def raiser(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Something went wrong")

    caplog.set_level("ERROR", "trend_analysis.data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "validate_market_data", raiser)
        frame = pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]})
        result = _validate_payload(
            frame,
            origin="payload",
            errors="log",
            include_date_column=True,
        )
    assert result is None
    assert "Unable to parse" not in caplog.text


def test_validate_payload_reraises_when_errors_set_to_raise() -> None:
    def raiser(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "validate_market_data", raiser)
        with pytest.raises(MarketDataValidationError):
            _validate_payload(
                pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]}),
                origin="payload",
                errors="raise",
                include_date_column=True,
            )


def test_load_csv_reads_file_and_applies_legacy_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "Date,Rate,Value\n2020-01-01,12%,100\n2020-01-02,10%,200\n", encoding="utf-8"
    )

    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)

    result = load_csv(
        str(csv_path),
        nan_policy="zeros",
        nan_limit="7",
    )

    assert result is not None
    assert list(result.columns) == ["Date", "Rate", "Value"]
    assert probe.calls[-1]["missing_policy"] == "zero"
    assert probe.calls[-1]["missing_limit"] == 7


def test_load_csv_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    assert load_csv(str(missing)) is None
    with pytest.raises(FileNotFoundError):
        load_csv(str(missing), errors="raise")


def test_load_parquet_uses_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")

    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)
    monkeypatch.setattr(pd, "read_parquet", lambda *_: pd.DataFrame({"A": [1, 2]}))

    result = load_parquet(str(parquet_path), include_date_column=False)

    assert result is not None
    assert probe.calls[-1]["source"].endswith("data.parquet")


def test_validate_dataframe_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1]})

    result = validate_dataframe(df)

    assert result is not None
    assert probe.calls[-1]["source"] == "dataframe"


def test_identify_risk_free_fund_returns_lowest_volatility() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "FundA": [1.0, 1.1, 1.2],
            "FundB": [1.0, 1.0, 1.0],
            "NonNumeric": ["x", "y", "z"],
        }
    )
    assert identify_risk_free_fund(df) == "FundB"


def test_identify_risk_free_fund_returns_none_when_no_numeric() -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-01", periods=2, freq="D"), "Label": ["a", "b"]}
    )
    assert identify_risk_free_fund(df) is None


def test_ensure_datetime_converts_and_rejects_bad_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"Date": ["01/02/20", "01/03/20"]})
    converted = ensure_datetime(frame.copy())
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])

    bad = pd.DataFrame({"Date": ["bad", "01/03/20"]})
    caplog.set_level("ERROR", "trend_analysis.data")
    with pytest.raises(ValueError):
        ensure_datetime(bad)
    assert "malformed date" in caplog.text.lower()


def test_normalise_numeric_strings_handles_percentages_and_parentheses() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "Pct": ["(12.5%)"],
            "Value": ["1,234"],
            "Already": [1.5],
        }
    )
    cleaned = _normalise_numeric_strings(frame)
    assert pytest.approx(cleaned.loc[0, "Pct"], rel=1e-6) == -0.125
    assert pytest.approx(cleaned.loc[0, "Value"], rel=1e-6) == 1234.0
    assert cleaned.loc[0, "Already"] == 1.5


def test_normalise_numeric_strings_leaves_non_numeric_columns() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-01"], "Label": ["abc"]})
    cleaned = _normalise_numeric_strings(frame)
    assert cleaned.loc[0, "Label"] == "abc"


def test_is_readable_checks_permission_bits() -> None:
    assert data._is_readable(stat.S_IRUSR)
    assert not data._is_readable(stat.S_IWUSR)


def test_load_csv_logs_permission_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "perm.csv"
    csv_path.write_text("Date,Value\n2020-01-01,1\n", encoding="utf-8")
    caplog.set_level("ERROR", "trend_analysis.data")

    monkeypatch.setattr(data, "_is_readable", lambda _mode: False)
    probe = ValidationProbe()
    monkeypatch.setattr(data, "validate_market_data", probe)

    assert load_csv(str(csv_path)) is None
    assert "Permission denied" in caplog.text
    assert not probe.calls

    with pytest.raises(PermissionError):
        load_csv(str(csv_path), errors="raise")


def test_load_csv_handles_directories_and_empty_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    caplog.set_level("ERROR", "trend_analysis.data")

    assert load_csv(str(directory)) is None
    with pytest.raises(IsADirectoryError):
        load_csv(str(directory), errors="raise")

    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    def raise_empty(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("empty")

    monkeypatch.setattr(pd, "read_csv", raise_empty)
    assert load_csv(str(csv_path)) is None
    assert "empty" in caplog.text


def test_load_csv_logs_validation_errors_without_parse_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Date,Value\n2020-01-01,1\n", encoding="utf-8")

    def raiser(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("General failure")

    monkeypatch.setattr(data, "validate_market_data", raiser)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_csv(str(csv_path)) is None
    assert "Unable to parse" not in caplog.text


def test_load_csv_logs_validation_errors_with_parse_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "bad_parse.csv"
    csv_path.write_text("Date,Value\n2020-01-01,1\n", encoding="utf-8")

    def raiser(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Dates could not be parsed")

    monkeypatch.setattr(data, "validate_market_data", raiser)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_csv(str(csv_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_handles_validation_error_from_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "helper.csv"
    csv_path.write_text("Date,Value\n2020-01-01,1\n", encoding="utf-8")

    def raiser(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data, "_validate_payload", raiser)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_csv(str(csv_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_permission_and_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"")

    monkeypatch.setattr(
        pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Value": [1]})
    )

    monkeypatch.setattr(data, "_is_readable", lambda _mode: False)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_parquet(str(parquet_path)) is None
    with pytest.raises(PermissionError):
        load_parquet(str(parquet_path), errors="raise")

    monkeypatch.setattr(data, "_is_readable", lambda _mode: True)

    def raise_empty(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("empty")

    monkeypatch.setattr(pd, "read_parquet", raise_empty)
    assert load_parquet(str(parquet_path)) is None

    def raise_validation(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("No parse")

    monkeypatch.setattr(
        pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Value": [1]})
    )
    monkeypatch.setattr(data, "validate_market_data", raise_validation)
    assert load_parquet(str(parquet_path)) is None


def test_load_parquet_applies_legacy_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "legacy.parquet"
    parquet_path.write_bytes(b"")

    probe = ValidationProbe()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pd.DataFrame({"Date": ["2020-01-01"], "Value": [1]}),
    )
    monkeypatch.setattr(data, "validate_market_data", probe)
    monkeypatch.setattr(data, "_is_readable", lambda _mode: True)

    result = load_parquet(
        str(parquet_path),
        nan_policy={"Value": "BackFill"},
        nan_limit={"Value": "4"},
    )

    assert result is not None
    call = probe.calls[-1]
    assert call["missing_policy"] == {"Value": "ffill"}
    assert call["missing_limit"] == {"Value": 4}


def test_load_parquet_handles_missing_and_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.parquet"
    assert load_parquet(str(missing)) is None
    with pytest.raises(FileNotFoundError):
        load_parquet(str(missing), errors="raise")

    directory = tmp_path / "dir"
    directory.mkdir()
    assert load_parquet(str(directory)) is None
    with pytest.raises(IsADirectoryError):
        load_parquet(str(directory), errors="raise")


def test_load_parquet_logs_validation_errors_with_parse_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "parse.parquet"
    parquet_path.write_bytes(b"")

    def raise_validation(*_args: object, **_kwargs: object) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data, "_is_readable", lambda _mode: True)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Value": [1]})
    )
    monkeypatch.setattr(data, "validate_market_data", raise_validation)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_parquet(str(parquet_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_handles_validation_error_from_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_path = tmp_path / "helper.parquet"
    parquet_path.write_bytes(b"")

    def raiser(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data, "_is_readable", lambda _mode: True)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"Value": [1]})
    )
    monkeypatch.setattr(data, "_validate_payload", raiser)
    caplog.set_level("ERROR", "trend_analysis.data")
    assert load_parquet(str(parquet_path)) is None
    assert "Unable to parse Date values" in caplog.text


def test_ensure_datetime_handles_iso_strings() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"]})
    converted = ensure_datetime(frame)
    assert pd.api.types.is_datetime64_any_dtype(converted["Date"])
