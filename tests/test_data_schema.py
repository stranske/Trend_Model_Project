import io

import pandas as pd
import pytest

from trend_portfolio_app.data_schema import (DATE_COL, _validate_df,
                                             infer_benchmarks,
                                             load_and_validate_csv,
                                             load_and_validate_file)


def test_validate_df_basic():
    csv = "Date,A,B\n2020-02-01,3,4\n2020-01-01,1,2\n"
    df, meta = load_and_validate_csv(io.StringIO(csv))
    # index should be month-end timestamps
    expected = [
        pd.Timestamp("2020-01-31").to_period("M").to_timestamp("M", how="end"),
        pd.Timestamp("2020-02-29").to_period("M").to_timestamp("M", how="end"),
    ]
    assert list(df.index) == expected
    assert meta["original_columns"] == ["A", "B"]
    assert meta["n_rows"] == 2


def test_validate_df_errors():
    # missing Date column
    with pytest.raises(ValueError):
        _validate_df(pd.DataFrame({"A": [1]}))

    # duplicate columns
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1], "B": [2]})
    df.columns = ["Date", "A", "A"]
    with pytest.raises(ValueError):
        _validate_df(df)

    # all NA returns
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [float("nan")]})
    with pytest.raises(ValueError):
        _validate_df(df)


def test_load_and_validate_file_excel(tmp_path):
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "test.xlsx"
    df2, meta = load_and_validate_file(buf)
    assert DATE_COL not in df2.columns
    assert meta["n_rows"] == 1


def test_load_and_validate_file_seek_error():
    class NoSeek(io.StringIO):
        def seek(self, *args, **kwargs):
            raise RuntimeError("no seek")

    buf = NoSeek("Date,A\n2020-01-01,1\n")
    buf.name = "data.csv"
    df, meta = load_and_validate_file(buf)
    assert meta["n_rows"] == 1


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
