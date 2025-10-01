from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis import data as data_mod
from trend_analysis.io.market_data import MarketDataValidationError


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
