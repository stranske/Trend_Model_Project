import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pandas as pd
import pytest

from trend_analysis import data as data_mod
from trend_analysis.io.market_data import MarketDataValidationError, ValidatedMarketData


def make_validated(frame: pd.DataFrame, **metadata_fields: Any) -> ValidatedMarketData:
    metadata_defaults: dict[str, Any] = {
        "mode": SimpleNamespace(value="csv"),
        "frequency": "daily",
        "frequency_detected": "D",
        "frequency_label": "Daily",
        "frequency_median_spacing_days": 1.0,
        "frequency_missing_periods": 0,
        "frequency_max_gap_periods": 0,
        "frequency_tolerance_periods": 0,
        "columns": list(frame.columns),
        "rows": len(frame),
        "date_range": ("2024-01-01", "2024-01-02"),
        "missing_policy": "ffill",
        "missing_policy_limit": 1,
        "missing_policy_summary": "filled",
    }
    metadata_defaults.update(metadata_fields)
    metadata = SimpleNamespace(**metadata_defaults)
    return ValidatedMarketData(frame=frame, metadata=metadata)  # type: ignore[arg-type]


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    index = pd.Index(pd.date_range("2024-01-01", periods=3, freq="D"), name="Date")
    return pd.DataFrame(
        {
            "FundA": [1.0, 1.2, 1.1],
            "FundB": [0.9, 1.0, 1.05],
        },
        index=index,
    )


def test_normalise_policy_alias_variants() -> None:
    assert data_mod._normalise_policy_alias(None) == "drop"
    assert data_mod._normalise_policy_alias("  BOTH  ") == "ffill"
    assert data_mod._normalise_policy_alias("zeros") == "zero"
    assert data_mod._normalise_policy_alias(" custom ") == "custom"


def test_normalise_policy_alias_handles_empty_string() -> None:
    assert data_mod._normalise_policy_alias("   ") == "drop"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("none", None),
        ("10", 10),
        (10, 10),
    ],
)
def test_coerce_limit_entry_allows_null_and_positive(value: Any, expected: int | None) -> None:
    assert data_mod._coerce_limit_entry(value) == expected


def test_coerce_limit_entry_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        data_mod._coerce_limit_entry(-1)
    with pytest.raises(ValueError):
        data_mod._coerce_limit_entry("abc")


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("strict", "strict"),
        ({"*": "drop"}, {"*": "drop"}),
    ],
)
def test_coerce_policy_kwarg_accepts_strings_and_mappings(
    value: Any, expected: str | Mapping[str, str] | None
) -> None:
    assert data_mod._coerce_policy_kwarg(value) == expected


def test_coerce_policy_kwarg_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        data_mod._coerce_policy_kwarg(42)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ({"*": "3"}, {"*": "3"}),
        (5, 5),
        ("7", 7),
        (" none ", None),
    ],
)
def test_coerce_limit_kwarg_parses_common_inputs(value: Any, expected: Any) -> None:
    assert data_mod._coerce_limit_kwarg(value) == expected


def test_coerce_limit_kwarg_accepts_numeric_string() -> None:
    assert data_mod._coerce_limit_kwarg("42") == 42


def test_coerce_limit_kwarg_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        data_mod._coerce_limit_kwarg(3.5)


def test_normalise_numeric_strings_handles_percentages_and_negatives() -> None:
    frame = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Alpha": ["1,234", "(567)"],
        "Beta": ["10%", "5%"],
        "Gamma": ["text", ""],
    })
    cleaned = data_mod._normalise_numeric_strings(frame)
    assert cleaned["Alpha"].tolist() == [1234.0, -567.0]
    assert cleaned["Beta"].tolist() == [0.10, 0.05]
    assert cleaned["Gamma"].tolist() == ["text", ""]


def test_finalise_validated_frame_copies_metadata(sample_frame: pd.DataFrame) -> None:
    payload = make_validated(sample_frame, rows=3)
    result = data_mod._finalise_validated_frame(payload, include_date_column=False)
    assert result.equals(sample_frame)
    assert result.attrs["market_data"]["metadata"] is payload.metadata
    assert result.attrs["market_data_rows"] == 3


def test_finalise_validated_frame_resets_index(sample_frame: pd.DataFrame) -> None:
    payload = make_validated(sample_frame)
    result = data_mod._finalise_validated_frame(payload, include_date_column=True)
    assert "Date" in result.columns
    assert result.attrs["market_data_mode"] == payload.metadata.mode.value


def test_validate_payload_builds_policy_and_limit_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = pd.DataFrame({"Date": ["2024-01-01"], "Fund": ["1.0"]})

    captured: dict[str, Any] = {}

    def fake_validate(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        df = pd.DataFrame({"Fund": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
        df.index.name = "Date"
        return make_validated(df)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    result = data_mod._validate_payload(
        payload,
        origin="upload.csv",
        errors="log",
        include_date_column=True,
        missing_policy={"Fund": "BOTH", "*": "zeros"},
        missing_limit={"Fund": "5", "*": "none"},
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert captured["missing_policy"] == {"Fund": "ffill", "*": "zero"}
    assert captured["missing_limit"] == {"Fund": 5, "*": None}
    assert list(result.columns) == ["Date", "Fund"]


def test_validate_payload_handles_scalar_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = pd.DataFrame({"Date": ["2024-01-01"], "Fund": [1.0]})

    def fake_validate(*args: Any, **kwargs: Any):
        frame = pd.DataFrame({"Fund": [1.0]}, index=pd.Index([pd.Timestamp("2024-01-01")], name="Date"))
        return make_validated(frame)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    result = data_mod._validate_payload(
        payload,
        origin="manual",
        errors="log",
        include_date_column=False,
        missing_policy="drop",
        missing_limit="10",
    )

    assert isinstance(result, pd.DataFrame)


def test_validate_payload_logs_market_data_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    payload = pd.DataFrame({"Date": ["2024-01-01"], "Fund": [1.0]})

    def boom(*_: Any, **__: Any):
        raise MarketDataValidationError("Could not be parsed", ["bad date"])

    monkeypatch.setattr(data_mod, "validate_market_data", boom)

    caplog.set_level("ERROR")
    result = data_mod._validate_payload(
        payload,
        origin="upload.csv",
        errors="log",
        include_date_column=False,
    )
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_validate_payload_raises_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = pd.DataFrame({"Date": ["2024-01-01"], "Fund": [1.0]})

    def boom(*_: Any, **__: Any):
        raise MarketDataValidationError("bad news")

    monkeypatch.setattr(data_mod, "validate_market_data", boom)

    with pytest.raises(MarketDataValidationError):
        data_mod._validate_payload(payload, origin="upload.csv", errors="raise", include_date_column=True)


@pytest.mark.parametrize(
    "mode_bits, expected",
    [
        (stat.S_IRUSR, True),
        (stat.S_IRUSR | stat.S_IWUSR, True),
        (stat.S_IWUSR, False),
    ],
)
def test_is_readable_checks_mode_bits(mode_bits: int, expected: bool) -> None:
    assert data_mod._is_readable(mode_bits) is expected


def test_load_csv_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "market.csv"
    df = pd.DataFrame({"Date": ["2024-01-01"], "Fund": ["1.2"]})
    df.to_csv(csv_path, index=False)

    def fake_validate(payload: pd.DataFrame, **_: Any):
        index = pd.Index(pd.to_datetime(payload["Date"]), name="Date")
        fund = pd.to_numeric(payload["Fund"], errors="coerce")
        frame = pd.DataFrame({"Fund": fund.values}, index=index)
        return make_validated(frame)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    result = data_mod.load_csv(str(csv_path), include_date_column=False)
    assert isinstance(result, pd.DataFrame)
    assert "market_data_columns" in result.attrs
    assert list(result.columns) == ["Fund"]


def test_load_csv_includes_date_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "market.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "Fund": ["1.0"]}).to_csv(csv_path, index=False)

    def fake_validate(_: pd.DataFrame, **__: Any):
        frame = pd.DataFrame({"Fund": [1.0]}, index=pd.Index([pd.Timestamp("2024-01-01")], name="Date"))
        return make_validated(frame)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    result = data_mod.load_csv(str(csv_path), include_date_column=True)
    assert list(result.columns) == ["Date", "Fund"]


def test_load_csv_handles_permission_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "locked.csv"
    csv_path.write_text("Date,Fund\n2024-01-01,1.0\n")

    monkeypatch.setattr(data_mod, "_is_readable", lambda _: False)
    caplog.set_level("ERROR")

    result = data_mod.load_csv(str(csv_path))
    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_missing_file_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("ERROR")
    result = data_mod.load_csv("/missing/file.csv")
    assert result is None
    assert "/missing/file.csv" in caplog.text


def test_load_csv_raises_when_requested(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        data_mod.load_csv(str(csv_path), errors="raise")


def test_load_csv_logs_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("Date,Fund\n2024-01-01,1.0\n")

    def boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", boom)
    caplog.set_level("ERROR")
    result = data_mod.load_csv(str(csv_path))
    assert result is None
    assert "Unexpected error loading" in caplog.text


def test_load_parquet_permission_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "market.parquet"
    pq_path.write_text("placeholder")

    class DummyStat:
        st_mode = 0

    monkeypatch.setattr(Path, "exists", lambda self: self == pq_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "stat", lambda self: DummyStat())

    with pytest.raises(PermissionError):
        data_mod.load_parquet(str(pq_path), errors="raise")


def test_load_parquet_validation_error_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pq_path = tmp_path / "market.parquet"
    pq_path.write_text("placeholder")

    monkeypatch.setattr(Path, "exists", lambda self: self == pq_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_mode=stat.S_IRUSR))
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame({"Date": ["2024-01-01"], "Fund": ["1.0"]}))

    def boom(*_: Any, **__: Any):
        raise MarketDataValidationError("Unable to parse date")

    monkeypatch.setattr(data_mod, "validate_market_data", boom)

    caplog.set_level("ERROR")
    result = data_mod.load_parquet(str(pq_path))
    assert result is None
    assert "Unable to parse Date values" in caplog.text


def test_load_parquet_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "market.parquet"
    pq_path.write_text("placeholder")

    monkeypatch.setattr(Path, "exists", lambda self: self == pq_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_mode=stat.S_IRUSR))
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame({"Date": ["2024-01-01"], "Fund": [1.0]}))

    def fake_validate(_: pd.DataFrame, **__: Any):
        frame = pd.DataFrame({"Fund": [1.0]}, index=pd.Index([pd.Timestamp("2024-01-01")], name="Date"))
        return make_validated(frame)

    monkeypatch.setattr(data_mod, "validate_market_data", fake_validate)

    result = data_mod.load_parquet(str(pq_path), include_date_column=True)
    assert list(result.columns) == ["Date", "Fund"]


def test_load_parquet_logs_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    pq_path = tmp_path / "market.parquet"
    pq_path.write_text("placeholder")

    monkeypatch.setattr(Path, "exists", lambda self: self == pq_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_mode=stat.S_IRUSR))
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    caplog.set_level("ERROR")
    result = data_mod.load_parquet(str(pq_path))
    assert result is None
    assert "Unexpected error loading" in caplog.text


def test_validate_dataframe_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"Date": ["2024-01-01"], "Fund": [1.0]})
    captured: dict[str, Any] = {}

    def fake_validate(
        payload: pd.DataFrame,
        *,
        origin: str,
        errors: str,
        include_date_column: bool,
    ) -> pd.DataFrame:
        captured["origin"] = origin
        captured["errors"] = errors
        captured["include_date_column"] = include_date_column
        index = pd.Index(pd.to_datetime(payload["Date"]), name="Date")
        frame = pd.DataFrame({"Fund": payload["Fund"].values}, index=index)
        return frame

    monkeypatch.setattr(data_mod, "_validate_payload", fake_validate)

    result = data_mod.validate_dataframe(df, include_date_column=False, origin="manual")
    assert isinstance(result, pd.DataFrame)
    assert captured == {
        "origin": "manual",
        "errors": "log",
        "include_date_column": False,
    }
    assert list(result.columns) == ["Fund"]


def test_identify_risk_free_fund_prefers_lowest_volatility() -> None:
    class DummyNumericFrame:
        @staticmethod
        def std(skipna: bool = True) -> SimpleNamespace:
            return SimpleNamespace(idxmin=lambda: "Cash")

    class DummyFrame:
        @staticmethod
        def select_dtypes(_: str) -> SimpleNamespace:
            return SimpleNamespace(columns=["Cash", "Aggressive"])

        def __getitem__(self, cols: list[str]) -> DummyNumericFrame:
            assert cols == ["Cash", "Aggressive"]
            return DummyNumericFrame()

    choice = data_mod.identify_risk_free_fund(DummyFrame())
    assert choice == "Cash"


def test_identify_risk_free_fund_returns_none_for_non_numeric() -> None:
    class DummyFrame:
        @staticmethod
        def select_dtypes(_: str) -> SimpleNamespace:
            return SimpleNamespace(columns=[])

    assert data_mod.identify_risk_free_fund(DummyFrame()) is None


def test_ensure_datetime_parses_specific_format() -> None:
    df = pd.DataFrame({"Date": ["01/02/24"]})
    result = data_mod.ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_ensure_datetime_raises_on_malformed_dates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeMask:
        def __init__(self, mask: list[bool]) -> None:
            self._mask = mask

        def any(self) -> bool:
            return any(self._mask)

        def sum(self) -> int:
            return sum(self._mask)

        def __iter__(self):
            return iter(self._mask)

    class FakeParsedDates:
        def __init__(self, raw: list[str]) -> None:
            self._raw = raw

        def isna(self) -> FakeMask:
            mask = [value == "bad" for value in self._raw]
            return FakeMask(mask)

    class FakeLoc:
        def __init__(self, values: list[str]) -> None:
            self._values = values

        def __getitem__(self, key: tuple[Iterable[bool], str]) -> SimpleNamespace:
            mask, _ = key
            selected = [value for value, flag in zip(self._values, mask) if flag]
            return SimpleNamespace(to_list=lambda: selected, tolist=lambda: selected)

    class FakeFrame:
        def __init__(self, values: list[str]) -> None:
            self._column = values
            self.columns = ["Date"]
            self.loc = FakeLoc(values)

        def __getitem__(self, column: str) -> list[str]:
            return self._column

        def __setitem__(self, column: str, value: Any) -> None:
            self._column = list(value)

    frame = FakeFrame(["bad", "2024-01-01"])
    call_count = {"tries": 0}

    def fake_to_datetime(values: Any, format: str | None = None, errors: str = "raise"):
        call_count["tries"] += 1
        if call_count["tries"] == 1:
            raise ValueError("bad format")
        return FakeParsedDates(["bad", "good"])

    monkeypatch.setattr(pd, "to_datetime", fake_to_datetime)
    monkeypatch.setattr(data_mod, "is_datetime64_any_dtype", lambda _: False)
    caplog.set_level("ERROR")
    with pytest.raises(ValueError):
        data_mod.ensure_datetime(frame)  # type: ignore[arg-type]
    assert "malformed date" in caplog.text.lower()
