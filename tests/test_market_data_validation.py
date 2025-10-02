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
    assert "irregular sampling" in str(exc.value).lower()


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


def test_validate_market_data_allows_weekend_gap() -> None:
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, -0.005, 0.004, 0.003, -0.002, 0.001],
        }
    )
    validated = validate_market_data(df)
    meta = validated.attrs.get("market_data", {})
    assert meta["frequency_code"] == "D"
    assert meta["frequency_missing_periods"] >= 2
    assert meta["frequency_max_gap_periods"] == 2
    assert meta["frequency_tolerance_periods"] >= 3


def test_missing_policy_drops_sparse_columns() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.015, 0.03],
            "FundB": [0.01, None, None, 0.02],
        }
    )
    validated = validate_market_data(df, missing_policy="drop")
    assert list(validated.columns) == ["FundA"]
    meta = validated.attrs["market_data"]
    assert meta["missing_policy"] == "drop"
    assert meta["missing_policy_dropped"] == ["FundB"]
    assert "FundB" in meta["missing_policy_summary"]


def test_missing_policy_ffill_with_limit() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.015, 0.02, 0.025],
            "FundB": [None, 0.02, 0.025, None],
        }
    )
    validated = validate_market_data(df, missing_policy="ffill", missing_limit=2)
    assert list(validated.columns) == ["FundA", "FundB"]
    assert validated["FundB"].isna().sum() == 0
    meta = validated.attrs["market_data"]
    assert meta["missing_policy"] == "ffill"
    assert meta["missing_policy_filled"]["FundB"]["count"] == 2
    assert "FundB" in (meta["missing_policy_summary"] or "")


def test_missing_policy_respects_limit() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [
                0.01,
                None,
                None,
                None,
                0.015,
                0.02,
            ],
        }
    )
    with pytest.raises(MarketDataValidationError) as exc:
        validate_market_data(df, missing_policy="ffill", missing_limit=2)
    assert "policy" in str(exc.value).lower()


def test_missing_policy_per_column_overrides() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, None, 0.012, 0.013],
            "FundB": [0.02, None, 0.018, 0.017],
        }
    )
    policy = {"*": "drop", "FundB": "ffill"}
    limits = {"*": 0, "FundB": 1}
    validated = validate_market_data(df, missing_policy=policy, missing_limit=limits)
    assert list(validated.columns) == ["FundB"]
    meta = validated.attrs["market_data"]
    assert meta["missing_policy_overrides"] == {"FundB": "ffill"}
    assert meta["missing_policy_limits"]["FundB"] == 1
    assert meta["missing_policy_filled"]["FundB"]["count"] == 1
    assert meta["missing_policy_dropped"] == ["FundA"]


def test_missing_limit_extends_frequency_tolerance() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2024-01-31",
                "2024-02-29",
                "2024-05-31",
                "2024-06-30",
            ],
            "FundA": [0.01, 0.015, 0.02, 0.025],
        }
    )

    with pytest.raises(MarketDataValidationError):
        validate_market_data(df)

    validated = validate_market_data(df, missing_limit=2)
    meta = validated.attrs["market_data"]
    assert meta["frequency_missing_periods"] == 2
    assert meta["frequency_tolerance_periods"] == 2


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
