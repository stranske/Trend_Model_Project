from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trend_analysis import data as data_mod
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    MissingPolicyFillDetails,
    ValidatedMarketData,
)


def test_load_csv_success(tmp_path: Path) -> None:
    csv = tmp_path / "returns.csv"
    csv.write_text("Date,A,B\n2024-01-31,0.01,0.02\n2024-02-29,0.03,-0.01\n")

    df = data_mod.load_csv(str(csv))
    assert df is not None
    assert list(df.columns) == ["Date", "A", "B"]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df.attrs["market_data_mode"] == "returns"


def test_load_csv_returns_none_by_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv = tmp_path / "duplicate.csv"
    csv.write_text("Date,A\n2024-01-31,0.01\n2024-01-31,0.02\n")

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(csv))
    assert result is None
    assert "Duplicate timestamps" in caplog.text


def test_load_csv_raises_when_requested(tmp_path: Path) -> None:
    csv = tmp_path / "duplicate.csv"
    csv.write_text("Date,A\n2024-01-31,0.01\n2024-01-31,0.02\n")

    with pytest.raises(MarketDataValidationError) as exc:
        data_mod.load_csv(str(csv), errors="raise")
    assert "Duplicate" in str(exc.value)


def test_load_csv_numeric_normalisation(tmp_path: Path) -> None:
    csv = tmp_path / "coerce.csv"
    csv.write_text(
        """Date,Value,Percent,Neg
01/31/24,"1,234e-4",50%,(100e-3)
02/29/24,"2,468e-4",75%,(200e-3)
"""
    )

    df = data_mod.load_csv(str(csv))
    assert df is not None
    assert pytest.approx(df["Value"].tolist()) == [0.1234, 0.2468]
    assert df["Percent"].tolist() == [0.5, 0.75]
    assert pytest.approx(df["Neg"].tolist()) == [-0.1, -0.2]


def test_validate_dataframe_helper() -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    frame = pd.DataFrame({"Date": dates, "Fund": [0.01, 0.02, -0.01]})

    validated = data_mod.validate_dataframe(
        frame, include_date_column=False, errors="raise"
    )
    assert isinstance(validated.index, pd.DatetimeIndex)
    assert "market_data_mode" in validated.attrs


def test_identify_risk_free_fund_basic() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "A": [0.01, 0.02, 0.03],
            "B": [0.005, 0.004, 0.006],
        }
    )
    assert data_mod.identify_risk_free_fund(df) == "B"


def test_identify_risk_free_fund_no_numeric() -> None:
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-31"])})
    assert data_mod.identify_risk_free_fund(df) is None


def test_normalise_policy_alias_variants() -> None:
    assert data_mod._normalise_policy_alias(None) == "drop"
    assert data_mod._normalise_policy_alias("  BOTH  ") == "ffill"
    assert data_mod._normalise_policy_alias("zeros") == "zero"
    assert data_mod._normalise_policy_alias("custom") == "custom"


def test_normalise_policy_alias_blank_string() -> None:
    assert data_mod._normalise_policy_alias("   ") == "drop"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("4", 4),
        (4.0, 4),
        (None, None),
        ("none", None),
    ],
)
def test_coerce_limit_entry(value: Any, expected: int | None) -> None:
    assert data_mod._coerce_limit_entry(value) == expected


def test_coerce_limit_entry_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        data_mod._coerce_limit_entry("abc")
    with pytest.raises(ValueError):
        data_mod._coerce_limit_entry(-1)


def test_coerce_policy_kwarg_accepts_mapping() -> None:
    mapping = {"A": "FFill", "*": "zeros"}
    assert data_mod._coerce_policy_kwarg(mapping) == mapping


def test_coerce_policy_kwarg_none_returns_none() -> None:
    assert data_mod._coerce_policy_kwarg(None) is None


def test_coerce_policy_kwarg_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        data_mod._coerce_policy_kwarg(3.14)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value,expected",
    [
        (5, 5),
        (5.0, 5),
        ("7", 7),
        ("", None),
        (" none ", None),
        (None, None),
    ],
)
def test_coerce_limit_kwarg(value: Any, expected: int | None) -> None:
    assert data_mod._coerce_limit_kwarg(value) == expected


def test_coerce_limit_kwarg_mapping() -> None:
    mapping = {"A": 1, "B": None}
    assert data_mod._coerce_limit_kwarg(mapping) == mapping


def test_coerce_limit_kwarg_rejects_non_numeric_string() -> None:
    with pytest.raises(TypeError):
        data_mod._coerce_limit_kwarg("abc")


def test_coerce_limit_kwarg_invalid_string_branch() -> None:
    with pytest.raises(TypeError):
        data_mod._coerce_limit_kwarg("invalid")


def _build_metadata(columns: list[str]) -> MarketDataMetadata:
    base_kwargs = dict(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_label="monthly",
        start=pd.Timestamp("2024-01-31"),
        end=pd.Timestamp("2024-02-29"),
        rows=2,
        columns=columns,
        symbols=columns,
        missing_policy="ffill",
    )
    return MarketDataMetadata(**base_kwargs)


def test_normalise_numeric_strings_non_numeric() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-31", periods=2, freq="ME"),
            "Text": ["alpha", "beta"],
        }
    )

    result = data_mod._normalise_numeric_strings(frame)

    assert result["Text"].tolist() == ["alpha", "beta"]


def test_finalise_validated_frame_transfers_metadata() -> None:
    frame = pd.DataFrame(
        {"fund": [0.1, 0.2]},
        index=pd.DatetimeIndex(["2024-01-31", "2024-02-29"], name="Date"),
    )
    metadata = _build_metadata(["fund"])
    metadata.missing_policy_limit = 2
    metadata.missing_policy_summary = "filled"
    metadata.missing_policy_filled = {
        "fund": MissingPolicyFillDetails(method="ffill", count=1)
    }
    validated = ValidatedMarketData(frame=frame, metadata=metadata)

    result = data_mod._finalise_validated_frame(validated, include_date_column=True)
    assert list(result.columns) == ["Date", "fund"]
    attrs = result.attrs["market_data"]
    assert attrs["metadata"].missing_policy == "ffill"
    assert result.attrs["market_data_frequency_label"] == "monthly"
    assert result.attrs["market_data_missing_policy_limit"] == 2


def test_validate_payload_coerces_policy_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        recorded.update(kwargs)
        assert isinstance(kwargs["missing_policy"], dict)
        assert kwargs["missing_policy"]["A"] == "ffill"
        assert kwargs["missing_policy"]["B"] == "drop"
        assert kwargs["missing_policy"]["C"] == "2"
        assert kwargs["missing_policy"]["*"] == "drop"
        assert kwargs["missing_limit"]["B"] is None
        meta = _build_metadata(["A", "B", "C"])
        return ValidatedMarketData(frame=payload.set_index("Date"), metadata=meta)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    class LegacyMapping(dict):
        def __contains__(self, key: object) -> bool:  # pragma: no cover - simple shim
            if key == "*":
                return True
            return super().__contains__(key)

    payload = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-31", periods=2, freq="ME"),
            "A": [0.1, 0.2],
            "B": [0.3, 0.4],
            "C": [0.5, 0.6],
        }
    )

    result = data_mod._validate_payload(
        payload,
        origin="payload",
        errors="raise",
        include_date_column=False,
        missing_policy=LegacyMapping({"A": "BOTH", "B": None, "C": 2}),
        missing_limit={"A": "1", "B": None, "C": 3},
    )

    assert isinstance(result, pd.DataFrame)
    assert "Date" not in result.columns


def test_validate_payload_policy_without_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        captured.update(kwargs)
        meta = _build_metadata(["A"])
        return ValidatedMarketData(frame=payload.set_index("Date"), metadata=meta)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    payload = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=1, freq="ME"), "A": [0.1]}
    )

    data_mod._validate_payload(
        payload,
        origin="payload",
        errors="raise",
        include_date_column=True,
        missing_policy={"A": "drop"},
    )

    assert "*" not in captured["missing_policy"]


def test_validate_payload_scalar_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> ValidatedMarketData:
        assert kwargs["missing_limit"] == 4
        meta = _build_metadata(["A"])
        return ValidatedMarketData(frame=payload.set_index("Date"), metadata=meta)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    payload = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=1, freq="ME"), "A": [0.1]}
    )

    result = data_mod._validate_payload(
        payload,
        origin="payload",
        errors="raise",
        include_date_column=True,
        missing_limit="4",
    )

    assert list(result.columns) == ["Date", "A"]


def test_validate_payload_logs_validation_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    error = MarketDataValidationError("Could not be parsed")

    def raise_error(*_: Any, **__: Any) -> ValidatedMarketData:
        raise error

    monkeypatch.setattr(data_mod, "validate_market_data", raise_error)

    payload = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})
    with caplog.at_level("ERROR"):
        result = data_mod._validate_payload(
            payload,
            origin="payload.csv",
            errors="log",
            include_date_column=True,
        )
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_payload_reraises_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(*_: Any, **__: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("nope")

    monkeypatch.setattr(data_mod, "validate_market_data", raise_error)
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})

    with pytest.raises(MarketDataValidationError):
        data_mod._validate_payload(
            payload,
            origin="payload.csv",
            errors="raise",
            include_date_column=True,
        )


def test_is_readable_checks_mode_bits() -> None:
    readable_mode = stat.S_IRUSR | stat.S_IROTH
    non_readable_mode = stat.S_IWUSR | stat.S_IXUSR
    assert data_mod._is_readable(readable_mode)
    assert not data_mod._is_readable(non_readable_mode)


def test_load_csv_coerces_legacy_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv = tmp_path / "legacy.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    captured: dict[str, Any] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return payload

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.load_csv(
        str(csv),
        nan_policy="backfill",
        nan_limit="2",
        missing_limit=3,
    )

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_policy"] == "backfill"
    assert captured["missing_limit"] == 3


def test_load_csv_legacy_nan_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv = tmp_path / "legacy_nan.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    captured: dict[str, Any] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return payload

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.load_csv(str(csv), nan_limit="5")

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_limit"] == 5


def test_load_csv_permission_denied_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv = tmp_path / "secure.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(csv))

    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_permission_denied_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv = tmp_path / "secure.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)

    with pytest.raises(PermissionError):
        data_mod.load_csv(str(csv), errors="raise")


def test_load_csv_missing_file_raises_when_requested() -> None:
    with pytest.raises(FileNotFoundError):
        data_mod.load_csv("/nonexistent.csv", errors="raise")


def test_load_csv_logs_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "missing.csv"
    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(missing))
    assert result is None
    assert str(missing) in caplog.text


def test_load_csv_directory_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_load_csv_validation_error_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv = tmp_path / "invalid.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    def raise_error(*_: Any, **__: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data_mod, "_validate_payload", raise_error)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(csv))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_csv_validation_error_without_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv = tmp_path / "invalid2.csv"
    csv.write_text("Date,A\n2024-01-31,1.0\n")

    def raise_error(*_: Any, **__: Any) -> ValidatedMarketData:
        raise MarketDataValidationError("Generic failure")

    monkeypatch.setattr(data_mod, "_validate_payload", raise_error)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(csv))

    assert result is None
    assert "Unable to parse Date values" not in caplog.text


def test_load_csv_parser_error_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("Date,A\n1,2\n")

    def raise_parser_error(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise pd.errors.ParserError("bad parse")

    monkeypatch.setattr(pd, "read_csv", raise_parser_error)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(csv))

    assert result is None
    assert "bad parse" in caplog.text


def test_load_parquet_invokes_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_file = tmp_path / "data.parquet"
    parquet_file.write_bytes(b"")

    payload = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})

    monkeypatch.setattr(pd, "read_parquet", lambda _: payload)

    captured: dict[str, Any] = {}

    def fake_validate(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.load_parquet(str(parquet_file), missing_policy="zeros")

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_policy"] == "zeros"
    assert captured["include_date_column"] is True


def test_load_parquet_legacy_nan_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_file = tmp_path / "legacy.parquet"
    parquet_file.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame())

    captured: dict[str, Any] = {}

    def fake_validate(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.load_parquet(str(parquet_file), nan_limit="4")

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_limit"] == 4


def test_load_parquet_legacy_nan_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_file = tmp_path / "policy.parquet"
    parquet_file.write_bytes(b"")

    payload = pd.DataFrame(
        {"Date": pd.date_range("2024-01-31", periods=1, freq="ME"), "A": [0.1]}
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: payload)

    captured: dict[str, Any] = {}

    def fake_validate(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.load_parquet(str(parquet_file), nan_policy="BFill")

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_policy"] == "BFill"


def test_load_parquet_logs_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_file = tmp_path / "invalid.parquet"
    parquet_file.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame())

    def raise_error(*_: Any, **__: Any) -> pd.DataFrame:
        raise MarketDataValidationError("Could not be parsed")

    monkeypatch.setattr(data_mod, "_validate_payload", raise_error)

    with caplog.at_level("ERROR"):
        result = data_mod.load_parquet(str(parquet_file))

    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_validation_error_without_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    parquet_file = tmp_path / "invalid2.parquet"
    parquet_file.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame())

    def raise_error(*_: Any, **__: Any) -> pd.DataFrame:
        raise MarketDataValidationError("Generic failure")

    monkeypatch.setattr(data_mod, "_validate_payload", raise_error)

    with caplog.at_level("ERROR"):
        result = data_mod.load_parquet(str(parquet_file))

    assert result is None
    assert "Unable to parse Date values" not in caplog.text


def test_load_parquet_validation_error_raises_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_file = tmp_path / "invalid_raise.parquet"
    parquet_file.write_bytes(b"")

    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame())

    def raise_error(*_: Any, **__: Any) -> pd.DataFrame:
        raise MarketDataValidationError("invalid")

    monkeypatch.setattr(data_mod, "_validate_payload", raise_error)

    with pytest.raises(MarketDataValidationError):
        data_mod.load_parquet(str(parquet_file), errors="raise")


def test_load_parquet_logs_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "missing.parquet"

    with caplog.at_level("ERROR"):
        result = data_mod.load_parquet(str(missing))

    assert result is None
    assert str(missing) in caplog.text


def test_load_parquet_permission_error_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parquet_file = tmp_path / "data.parquet"
    parquet_file.write_bytes(b"")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _mode: False)

    with pytest.raises(PermissionError):
        data_mod.load_parquet(str(parquet_file), errors="raise")


def test_load_parquet_directory_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with caplog.at_level("ERROR"):
        result = data_mod.load_parquet(str(directory))

    assert result is None
    assert str(directory) in caplog.text


def test_validate_dataframe_passes_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})

    captured: dict[str, Any] = {}

    def fake_validate(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.validate_dataframe(
        payload, origin="in-memory", include_date_column=False
    )

    assert isinstance(result, pd.DataFrame)
    assert captured["origin"] == "in-memory"
    assert captured["include_date_column"] is False


def test_identify_risk_free_fund_logs_choice(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "A": [0.1, 0.1, 0.1],
            "B": [0.01, 0.02, 0.03],
        }
    )

    with caplog.at_level("INFO"):
        fund = data_mod.identify_risk_free_fund(frame)

    assert fund == "A"
    assert "Risk-free column" in caplog.text


def test_ensure_datetime_raises_on_malformed_dates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"Date": ["2024-01-31", "not-a-date"]})

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            data_mod.ensure_datetime(frame.copy())

    assert "malformed date" in caplog.text.lower()


def test_ensure_datetime_converts_strings() -> None:
    frame = pd.DataFrame({"Date": ["01/31/24", "02/29/24"]})

    result = data_mod.ensure_datetime(frame.copy())

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_skips_missing_column() -> None:
    frame = pd.DataFrame({"Other": [1, 2, 3]})

    result = data_mod.ensure_datetime(frame.copy())

    assert list(result.columns) == ["Other"]


def test_ensure_datetime_generic_parse() -> None:
    frame = pd.DataFrame({"Date": ["2024-01-31", "2024-02-29"]})

    result = data_mod.ensure_datetime(frame.copy())

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
