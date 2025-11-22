import io

import pandas as pd
import pytest

from trend_analysis.io.market_data import MarketDataMode, MarketDataValidationError
from trend_portfolio_app.data_schema import (
    DATE_COL,
    _validate_df,
    infer_benchmarks,
    load_and_validate_csv,
    load_and_validate_file,
)


def test_validate_df_basic():
    csv = "Date,A,B\n2020-01-01,0.01,0.02\n2020-02-01,0.03,0.04\n"
    df, meta = load_and_validate_csv(io.StringIO(csv))
    expected = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert list(df.index) == expected
    assert meta["original_columns"] == ["A", "B"]
    assert meta["n_rows"] == 2
    assert meta["metadata"].mode == MarketDataMode.RETURNS
    assert meta["frequency"] in {"monthly", "31D"}
    assert meta["frequency_code"] in {"M", "31D"}
    assert meta["symbols"] == ["A", "B"]
    assert meta["validation"]["issues"] == []
    assert any("Dataset is quite small" in w for w in meta["validation"]["warnings"])


def test_validate_df_errors():
    # missing Date column
    with pytest.raises(MarketDataValidationError):
        _validate_df(pd.DataFrame({"A": [1]}))

    # duplicate columns
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1], "B": [2]})
    df.columns = ["Date", "A", "A"]
    with pytest.raises(MarketDataValidationError):
        _validate_df(df)

    # all NA returns
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [float("nan")]})
    with pytest.raises(MarketDataValidationError):
        _validate_df(df)


def test_load_and_validate_file_excel(tmp_path):
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "A": [0.01, 0.02]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "test.xlsx"
    df2, meta = load_and_validate_file(buf)
    assert DATE_COL not in df2.columns
    assert meta["n_rows"] == 2
    assert meta["metadata"].mode == MarketDataMode.RETURNS
    assert meta["validation"]["issues"] == []


def test_load_and_validate_file_seek_error():
    class NoSeek(io.StringIO):
        def seek(self, *args, **kwargs):
            raise RuntimeError("no seek")

    buf = NoSeek("Date,A\n2020-01-01,0.01\n2020-02-01,0.02\n")
    buf.name = "data.csv"
    df, meta = load_and_validate_file(buf)
    assert meta["n_rows"] == 2


def test_load_and_validate_file_read_error():
    class BadFile:
        name = "bad.csv"

        def read(self, *args, **kwargs):  # pragma: no cover - via pandas
            raise ValueError("bad read")

    with pytest.raises(ValueError):
        load_and_validate_file(BadFile())


def test_infer_benchmarks():
    cols = ["SPX", "fund1", "MyIndex"]
    assert infer_benchmarks(cols) == ["SPX", "MyIndex"]


def test_validate_df_sanitizes_formula_headers():
    csv = "Date,=SUM(B1:B2),-Weird\n2020-01-01,0.01,0.02\n"
    df, meta = load_and_validate_csv(io.StringIO(csv))
    assert "SUM(B1:B2)" in df.columns
    assert "Weird" in df.columns
    sanitized = meta["sanitized_columns"]
    assert len(sanitized) == 2
    warnings = meta["validation"]["warnings"]
    assert any("Sanitized column headers" in warning for warning in warnings)


def test_validate_df_sanitizes_duplicate_formula_headers_unique():
    csv = "Date,=Alpha,=Alpha\n2020-01-01,0.01,0.02\n"
    df, meta = load_and_validate_csv(io.StringIO(csv))
    assert list(df.columns) == ["Alpha", "Alpha_2"]
    mapping = meta["sanitized_columns"]
    assert mapping[0]["sanitized"] == "Alpha"
    assert mapping[1]["sanitized"] == "Alpha_2"
