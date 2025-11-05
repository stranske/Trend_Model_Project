from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import data
from trend_analysis.io.market_data import MarketDataValidationError


def _validated_payload(frame: pd.DataFrame) -> SimpleNamespace:
    index = pd.DatetimeIndex(pd.to_datetime(frame["Date"]), name="Date")
    processed = frame.drop(columns=["Date"]).set_index(index)
    metadata = SimpleNamespace(
        mode=SimpleNamespace(value="returns"),
        frequency="D",
        frequency_detected="D",
        frequency_label="daily",
        frequency_median_spacing_days=1.0,
        frequency_missing_periods=0,
        frequency_max_gap_periods=0,
        frequency_tolerance_periods=0,
        columns=list(processed.columns),
        rows=len(processed),
        date_range=("2020-01-01", "2020-01-01"),
        missing_policy="drop",
        missing_policy_limit=None,
        missing_policy_summary="none",
    )
    return SimpleNamespace(frame=processed, metadata=metadata)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, data.DEFAULT_POLICY_FALLBACK),
        ("", data.DEFAULT_POLICY_FALLBACK),
        (" both ", "ffill"),
        ("zeros", "zero"),
        ("custom", "custom"),
    ],
)
def test_normalise_policy_alias_variants(raw: str | None, expected: str) -> None:
    assert data._normalise_policy_alias(raw) == expected


@pytest.mark.parametrize("value", [object(), [1], 42])
def test_coerce_policy_kwarg_rejects_invalid_types(value: object) -> None:
    with pytest.raises(TypeError):
        data._coerce_policy_kwarg(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("none", None), ("5", 5), (5.0, 5)],
)
def test_coerce_limit_entry_handles_strings(value: object, expected: int | None) -> None:
    assert data._coerce_limit_entry(value) == expected


def test_coerce_limit_entry_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        data._coerce_limit_entry(-1)


def test_coerce_limit_kwarg_accepts_numeric_strings() -> None:
    assert data._coerce_limit_kwarg("10") == 10
    assert data._coerce_limit_kwarg(" none ") is None


def test_normalise_numeric_strings_handles_percent_and_commas() -> None:
    frame = pd.DataFrame({
        "Date": ["2020-01-01"],
        "FundA": ["12.5%"],
        "FundB": ["(1,200)"]
    })
    cleaned = data._normalise_numeric_strings(frame)
    assert pytest.approx(cleaned.loc[0, "FundA"], rel=1e-9) == 0.125
    assert cleaned.loc[0, "FundB"] == -1200


def test_validate_payload_normalises_policies_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({
        "Date": ["2020-01-01"],
        "FundA": ["50%"],
    })
    captured: dict[str, object] = {}

    def fake_validate(payload: pd.DataFrame, **kwargs: object):  # noqa: ANN003
        captured["payload"] = payload
        captured.update(kwargs)
        return _validated_payload(frame.assign(FundA=[0.5]))

    monkeypatch.setattr(data, "validate_market_data", fake_validate)
    result = data._validate_payload(
        frame,
        origin="demo",
        errors="raise",
        include_date_column=False,
        missing_policy={"FundA": "BOTH", "*": None},
        missing_limit={"FundA": "5", "*": "none"},
    )

    assert isinstance(result, pd.DataFrame)
    assert captured["missing_policy"] == {"FundA": "ffill", "*": data.DEFAULT_POLICY_FALLBACK}
    assert captured["missing_limit"] == {"FundA": 5, "*": None}
    payload = captured["payload"]
    assert pytest.approx(payload.loc[0, "FundA"], rel=1e-9) == 0.5


def test_validate_payload_logs_parse_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1.0]})

    def fail_validate(*_: object, **__: object) -> None:  # noqa: ANN002, ANN003
        raise MarketDataValidationError("Date column could not be parsed")

    monkeypatch.setattr(data, "validate_market_data", fail_validate)
    with caplog.at_level(logging.ERROR, logger="trend_analysis.data"):
        result = data._validate_payload(
            frame,
            origin="source.csv",
            errors="log",
            include_date_column=True,
        )
    assert result is None
    assert "Unable to parse Date values in source.csv" in caplog.text


def test_validate_dataframe_enriches_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1.0]})

    def fake_validate(payload: pd.DataFrame, **kwargs: object):  # noqa: ANN003
        return _validated_payload(frame)

    monkeypatch.setattr(data, "validate_market_data", fake_validate)
    validated = data.validate_dataframe(frame, origin="inline")
    assert isinstance(validated, pd.DataFrame)
    assert list(validated.columns) == ["Date", "FundA"]
    attrs = validated.attrs
    assert attrs["market_data"]["metadata"].missing_policy == "drop"
    assert attrs["market_data_frequency_label"] == "daily"


def test_load_csv_missing_file_logs_error(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with caplog.at_level(logging.ERROR, logger="trend_analysis.data"):
        assert data.load_csv(str(missing)) is None
    assert str(missing) in caplog.text


def test_load_csv_raises_when_configured(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        data.load_csv(str(missing), errors="raise")


def test_load_parquet_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "dataset.parquet"
    file_path.write_bytes(b"")
    frame = pd.DataFrame({"Date": ["2020-01-01"], "FundA": [1.0]})

    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(data, "validate_market_data", lambda payload, **_: _validated_payload(payload))

    result = data.load_parquet(str(file_path), include_date_column=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["FundA"]


def test_identify_risk_free_fund_selects_lowest_std() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "FundA": [1.0, 1.1, 0.9, 1.05],
            "FundB": [2.0, 3.0, 4.0, 5.0],
        }
    )
    assert data.identify_risk_free_fund(frame) == "FundA"


def test_identify_risk_free_fund_returns_none_when_empty() -> None:
    frame = pd.DataFrame({"Date": ["2020-01-01"], "Text": ["n/a"]})
    assert data.identify_risk_free_fund(frame) is None


def test_ensure_datetime_parses_valid_strings() -> None:
    frame = pd.DataFrame({"Date": ["01/02/20"]})
    parsed = data.ensure_datetime(frame.copy())
    assert pd.api.types.is_datetime64_any_dtype(parsed["Date"])


def test_ensure_datetime_reports_invalid_values(caplog: pytest.LogCaptureFixture) -> None:
    frame = pd.DataFrame({"Date": ["not-a-date", "still bad"]})
    with caplog.at_level(logging.ERROR, logger="trend_analysis.data"):
        with pytest.raises(ValueError):
            data.ensure_datetime(frame)
    assert "malformed date(s)" in caplog.text.lower()
