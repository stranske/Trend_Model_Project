import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataValidationError,
    validate_market_data,
)


def _build_returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    data = {
        "Date": dates,
        "FundA": [0.01, 0.02, -0.015, 0.005],
        "FundB": [0.03, -0.01, 0.0, 0.012],
    }
    return pd.DataFrame(data)


def test_validate_market_data_happy_path_returns() -> None:
    df = _build_returns_frame()
    validated = validate_market_data(df)

    assert isinstance(validated.index, pd.DatetimeIndex)
    assert validated.index.name == "Date"
    assert list(validated.columns) == ["FundA", "FundB"]
    meta = validated.attrs.get("market_data", {})
    assert meta["mode"] == "returns"
    assert meta["frequency"] == "monthly"
    assert pd.Timestamp(meta["start"]) == validated.index.min()
    assert pd.Timestamp(meta["end"]) == validated.index.max()
    assert meta["symbols"] == ["FundA", "FundB"]


def test_validate_market_data_duplicate_dates() -> None:
    df = _build_returns_frame()
    df.loc[3, "Date"] = df.loc[2, "Date"]

    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)
    assert "Duplicate timestamps" in str(exc.value)


def test_validate_market_data_unsorted_dates() -> None:
    df = _build_returns_frame().iloc[[2, 0, 1, 3]].reset_index(drop=True)

    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)
    assert "sorted in ascending order" in str(exc.value)
    assert exc.value.issues
    assert any("sorted" in issue for issue in exc.value.issues)


def test_validate_market_data_mixed_frequency() -> None:
    df = _build_returns_frame()
    df.loc[2, "Date"] = pd.Timestamp("2024-03-15")

    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)
    assert "Mixed sampling cadence" in str(exc.value)


def test_validate_market_data_price_mode_detection() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Asset": [100.0, 101.5, 102.2, 101.9, 103.4],
        }
    )
    validated = validate_market_data(df)
    meta = validated.attrs.get("market_data", {})
    assert meta["mode"] == "prices"
    assert meta["frequency"] in {"daily", "business-daily"}
    assert meta["symbols"] == ["Asset"]


def test_validate_market_data_mixed_modes_detected() -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Fund": [0.01, -0.02, 0.015],
            "Index": [100.0, 101.0, 99.5],
        }
    )
    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)
    msg = str(exc.value)
    assert "mix of returns-like and price-like" in msg


def test_validate_market_data_ambiguous_mode() -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Signal": [2.0, 2.0, 2.0],
        }
    )
    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)
    assert "Unable to determine" in str(exc.value)
    assert exc.value.issues
    assert any("Unable to determine" in issue for issue in exc.value.issues)


def test_validate_market_data_missing_date_column_reports_issue() -> None:
    df = pd.DataFrame({"FundA": [0.01, 0.02, 0.03]})

    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df)

    assert exc.value.issues
    assert any("Missing a 'Date'" in issue for issue in exc.value.issues)


def test_validate_market_data_accepts_datetime_index() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    frame = pd.DataFrame({"FundA": [0.01, 0.02, 0.0, -0.01]})
    frame.index = dates
    validated = validate_market_data(frame)
    assert validated.index.equals(dates)
