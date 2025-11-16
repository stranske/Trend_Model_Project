import pandas as pd
import pytest

from trend.validation import (
    PRICE_SCHEMA,
    assert_execution_lag,
    enforce_required_columns,
    validate_prices_frame,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["FundA", "FundA", "FundB", "FundB"],
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-01-31",
                    "2020-02-29",
                ]
            ),
            "close": [1.0, 1.1, 2.0, 2.1],
        }
    )


def test_enforce_required_columns_missing_column() -> None:
    frame = _valid_frame().drop(columns=["close"])

    with pytest.raises(ValueError, match="missing required columns"):
        enforce_required_columns(frame, PRICE_SCHEMA)


def test_enforce_required_columns_wrong_dtype() -> None:
    frame = _valid_frame().copy()
    frame["close"] = frame["close"].astype(str)

    with pytest.raises(ValueError, match="column dtypes differ"):
        enforce_required_columns(frame, PRICE_SCHEMA)


def test_validate_prices_frame_rejects_unsorted_dates() -> None:
    frame = _valid_frame().copy()
    frame.loc[[0, 1], "date"] = frame.loc[[1, 0], "date"].values

    with pytest.raises(ValueError, match="within each symbol"):
        validate_prices_frame(frame)


def test_validate_prices_frame_rejects_duplicate_symbol_dates() -> None:
    frame = _valid_frame().copy()
    frame.loc[1, "date"] = frame.loc[0, "date"]

    with pytest.raises(ValueError, match="duplicate symbol/date"):
        validate_prices_frame(frame)


def test_assert_execution_lag_raises_on_stale_data() -> None:
    validated = validate_prices_frame(_valid_frame())

    with pytest.raises(ValueError, match="Price data stale"):
        assert_execution_lag(validated, as_of="2020-03-10", max_lag_days=5)
