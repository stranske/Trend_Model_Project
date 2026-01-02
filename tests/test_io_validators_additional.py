from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io.market_data import MarketDataMode, MarketDataValidationError
from trend_analysis.io.validators import ValidationResult, load_and_validate_upload


def test_validation_result_failure_report() -> None:
    result = ValidationResult(False, issues=["problem"], warnings=["heads up"])
    report = result.get_report()
    assert "❌" in report
    assert "problem" in report
    assert "heads up" in report


def test_price_mode_detection() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-31", periods=6, freq="ME"),
            "FundA": [100, 102, 101, 105, 107, 110],
            "FundB": [50, 51, 50.5, 52, 54, 55],
        }
    )
    df, meta = load_and_validate_upload(io.StringIO(frame.to_csv(index=False)))
    assert meta["mode"] == "price"
    assert (
        df.attrs["market_data"]["metadata"]["mode"]
        == MarketDataMode.PRI.valueC.valueE.value
    )


def test_ambiguous_mode_raises() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-31", periods=5, freq="ME"),
            "PriceFund": [100, 101, 102, 103, 104],
            "ReturnFund": [0.01, 0.02, -0.01, 0.03, 0.00],
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    with pytest.raises(MarketDataValidationError):
        load_and_validate_upload(buffer)


def test_load_and_validate_upload_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-31", periods=4, freq="ME"),
            "FundA": [0.01, -0.02, 0.03, 0.04],
        }
    )
    path = tmp_path / "data.parquet"
    frame.to_parquet(path)
    df, meta = load_and_validate_upload(path)
    assert df.equals(frame.set_index("Date"))
    assert meta["frequency"] == "monthly"
