from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from data.contracts import coerce_to_utc, validate_prices


def _price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"price": [101.0 + i for i in range(len(index))]}, index=index)
    frame.attrs["market_data_mode"] = "price"
    return frame


def test_validate_prices_happy_path() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz=timezone.utc)
    frame = _price_frame(idx)
    validated = validate_prices(frame, freq="D")
    assert validated.index.tz is timezone.utc


def test_validate_prices_rejects_unsorted_index() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-03", tz=timezone.utc),
            pd.Timestamp("2024-01-01", tz=timezone.utc),
        ]
    )
    frame = _price_frame(idx)
    with pytest.raises(ValueError, match="sorted in ascending order"):
        validate_prices(frame)


def test_validate_prices_rejects_duplicates() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01", tz=timezone.utc),
            pd.Timestamp("2024-01-01", tz=timezone.utc),
        ]
    )
    frame = _price_frame(idx)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(frame)


def test_validate_prices_requires_timezone() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    frame = _price_frame(idx.tz_localize(None))
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_prices(frame)


def test_validate_prices_detects_mixed_frequency() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01", tz=timezone.utc),
            pd.Timestamp("2024-01-02", tz=timezone.utc),
            pd.Timestamp("2024-01-04", tz=timezone.utc),
        ]
    )
    frame = _price_frame(idx)
    with pytest.raises(ValueError, match="mixed frequencies"):
        validate_prices(frame, freq="D")


def test_validate_prices_rejects_negative_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz=timezone.utc)
    frame = _price_frame(idx)
    frame.iloc[1, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_prices(frame)


def test_validate_prices_ignores_returns_mode_for_negatives() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz=timezone.utc)
    frame = pd.DataFrame({"ret": [0.1, -0.2]}, index=idx)
    frame.attrs["market_data_mode"] = "returns"
    validate_prices(frame)


def test_coerce_to_utc_sets_index_and_column() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "price": [101.0, 102.0],
        }
    )
    frame.attrs["market_data_mode"] = "price"
    coerced = coerce_to_utc(frame)
    assert isinstance(coerced.index, pd.DatetimeIndex)
    assert coerced.index.tz is timezone.utc
    assert str(coerced["Date"].dtype) == "datetime64[ns, UTC]"


def test_coerce_to_utc_converts_existing_timezones() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern")
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=idx)
    frame.attrs["market_data_mode"] = "price"
    coerced = coerce_to_utc(frame)
    assert coerced.index.tz is timezone.utc
    assert coerced.index[0] == pd.Timestamp("2024-01-01 05:00:00+0000", tz=timezone.utc)
