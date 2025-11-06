"""Additional coverage for ``trend_analysis.data`` helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis.data import (
    DEFAULT_POLICY_FALLBACK,
    _coerce_limit_kwarg,
    _validate_payload,
    load_csv,
    load_parquet,
)
from trend_analysis.io.market_data import MarketDataValidationError


class _WildcardMapping(dict[str, str]):
    """Dictionary that advertises a wildcard default without storing it."""

    def __contains__(self, item: object) -> bool:
        if item == "*":
            return True
        return super().__contains__(item)


def _dummy_finalise(validated: Any, *, include_date_column: bool) -> object:
    """Return a simple marker to avoid relying on real metadata structures."""

    assert hasattr(validated, "frame")
    assert include_date_column is False
    return SimpleNamespace(frame=validated.frame)


@pytest.mark.parametrize("value", ["none", "", "  none  "])
def test_coerce_limit_kwarg_handles_empty_strings(value: str) -> None:
    """String placeholders representing ``None`` should normalise to ``None``."""

    assert _coerce_limit_kwarg(value) is None


def test_coerce_limit_kwarg_accepts_numeric_strings() -> None:
    """Digit strings should be converted to integers without raising."""

    assert _coerce_limit_kwarg("15") == 15


def test_validate_payload_injects_wildcard_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mappings that advertise a wildcard should gain the fallback entry."""

    frame = pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=3), "A": [1, 2, 3]}
    )
    captured: dict[str, Any] = {}

    def fake_validate(
        payload: pd.DataFrame,
        *,
        source: str,
        missing_policy: Any,
        missing_limit: Any,
    ) -> Any:
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        return SimpleNamespace(frame=payload, metadata=SimpleNamespace())

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    monkeypatch.setattr(
        "trend_analysis.data._finalise_validated_frame", _dummy_finalise
    )

    mapping = _WildcardMapping({"A": "ffill"})

    result = _validate_payload(
        frame,
        origin="unit-test",
        errors="log",
        include_date_column=False,
        missing_policy=mapping,
        missing_limit=None,
    )

    assert isinstance(result, SimpleNamespace)
    assert captured["policy"]["*"] == DEFAULT_POLICY_FALLBACK
    assert captured["limit"] is None


def test_load_csv_missing_limit_kwarg_converted(
    tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_csv`` should coerce legacy ``missing_limit`` kwargs to integers."""

    path = tmp_path / "data.csv"
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "A": [1, 2]})
    frame.to_csv(path, index=False)

    captured: dict[str, Any] = {}

    def fake_validate(
        payload: pd.DataFrame,
        *,
        source: str,
        missing_policy: Any,
        missing_limit: Any,
    ) -> Any:
        captured["limit"] = missing_limit
        return SimpleNamespace(frame=payload, metadata=SimpleNamespace())

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    def finalise(validated: Any, *, include_date_column: bool) -> object:
        assert include_date_column is True
        return "finalised"

    monkeypatch.setattr("trend_analysis.data._finalise_validated_frame", finalise)

    result = load_csv(
        str(path),
        missing_limit="7",
        include_date_column=True,
        errors="log",
    )

    assert result == "finalised"
    assert captured["limit"] == 7


def test_load_csv_logs_validation_error_hint(
    tmp_path: pytest.Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors mentioning unparseable dates should include the friendly hint."""

    path = tmp_path / "data.csv"
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "A": [1, 2]})
    frame.to_csv(path, index=False)

    def fake_validate(*_: Any, **__: Any) -> Any:
        raise MarketDataValidationError("Dates could not be parsed", [])

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    caplog.set_level(logging.ERROR)

    result = load_csv(str(path), errors="log")

    assert result is None
    assert any(
        "Unable to parse Date values" in record.message and str(path) in record.message
        for record in caplog.records
    )


def test_load_parquet_missing_limit_kwarg_converted(
    tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_parquet`` should reuse the legacy missing-limit conversion logic."""

    path = tmp_path / "data.parquet"
    path.write_bytes(b"")
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2), "A": [1, 2]})

    monkeypatch.setattr("trend_analysis.data.pd.read_parquet", lambda _: frame)

    captured: dict[str, Any] = {}

    def fake_validate(
        payload: pd.DataFrame,
        *,
        source: str,
        missing_policy: Any,
        missing_limit: Any,
    ) -> Any:
        captured["limit"] = missing_limit
        return SimpleNamespace(frame=payload, metadata=SimpleNamespace())

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)
    monkeypatch.setattr(
        "trend_analysis.data._finalise_validated_frame", _dummy_finalise
    )

    result = load_parquet(str(path), missing_limit="9", include_date_column=False)

    assert isinstance(result, SimpleNamespace)
    assert captured["limit"] == 9


def test_load_parquet_logs_validation_error_hint(
    tmp_path: pytest.Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parquet loader should mirror CSV error messaging for date issues."""

    path = tmp_path / "data.parquet"
    path.write_bytes(b"")
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2), "A": [1, 2]})

    monkeypatch.setattr("trend_analysis.data.pd.read_parquet", lambda _: frame)

    def fake_validate(*_: Any, **__: Any) -> Any:
        raise MarketDataValidationError("unable to parse date column", [])

    monkeypatch.setattr("trend_analysis.data.validate_market_data", fake_validate)

    caplog.set_level(logging.ERROR)

    result = load_parquet(str(path), errors="log")

    assert result is None
    assert any(
        "Unable to parse Date values" in record.message and str(path) in record.message
        for record in caplog.records
    )
