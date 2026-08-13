import pandas as pd
import pytest

from trend_analysis.data import load_parquet
from trend_analysis.io.market_data import MarketDataValidationError


def _write_parquet(tmp_path, column: str) -> str:
    path = tmp_path / "returns.parquet"
    pd.DataFrame(
        {
            column: pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "ManagerA": [0.01, 0.03],
        }
    ).to_parquet(path)
    return str(path)


def test_load_parquet_honors_configured_date_column(tmp_path) -> None:
    frame = load_parquet(
        _write_parquet(tmp_path, "Timestamp"),
        errors="raise",
        date_column="Timestamp",
    )

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA"]
    assert pd.api.types.is_datetime64_any_dtype(frame["Date"])


def test_load_parquet_default_date_column_unchanged(tmp_path) -> None:
    frame = load_parquet(_write_parquet(tmp_path, "Date"), errors="raise")

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA"]


def test_load_parquet_rejects_missing_configured_date_column(tmp_path) -> None:
    with pytest.raises(MarketDataValidationError, match="Timestamp"):
        load_parquet(
            _write_parquet(tmp_path, "Date"),
            errors="raise",
            date_column="Timestamp",
        )
