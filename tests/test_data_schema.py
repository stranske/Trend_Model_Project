import io

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
    MissingPolicyFillDetails,
    ValidatedMarketData,
)
from trend_portfolio_app.data_schema import (
    DATE_COL,
    _build_meta,
    _sanitize_formula_headers,
    _validate_df,
    apply_original_headers,
    extract_headers_from_bytes,
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


def test_extract_headers_handles_edge_cases():
    assert extract_headers_from_bytes(b"", is_excel=False) == []
    assert extract_headers_from_bytes(b"", is_excel=True) == []
    # Malformed Excel payload yields None rather than raising
    assert extract_headers_from_bytes(b"not-excel", is_excel=True) is None


def test_apply_original_headers_mismatch():
    df = pd.DataFrame([[1, 2]], columns=["A", "B"])
    result = apply_original_headers(df, ["A"])
    assert result is None
    assert list(df.columns) == ["A", "B"]


def test_sanitize_formula_headers_no_changes():
    df = pd.DataFrame([[1, 2]], columns=["safe", "header"])
    sanitized, changes = _sanitize_formula_headers(df)
    assert sanitized is df
    assert changes == []


def test_build_meta_populates_warnings_and_metadata_fields():
    metadata = MarketDataMetadata(
        mode=MarketDataMode.PRICE,
        frequency="D",
        frequency_detected="D",
        frequency_label="daily",
        frequency_missing_periods=2,
        frequency_max_gap_periods=3,
        frequency_tolerance_periods=1,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-05"),
        rows=5,
        columns=["=Bad", "Index"],
        missing_policy="drop",
        missing_policy_limit=1,
        missing_policy_summary="dropped missing",
        missing_policy_dropped=["Index"],
        missing_policy_filled={
            "=Bad": MissingPolicyFillDetails(method="ffill", count=1)
        },
    )
    validated = ValidatedMarketData(
        pd.DataFrame({"=Bad": [1, None, None, None, None], "Index": [None] * 5}),
        metadata,
    )
    meta = _build_meta(
        validated, sanitized_columns=[{"original": "=Bad", "sanitized": "Bad"}]
    )

    warnings = meta["validation"]["warnings"]
    assert any("Dataset is quite small" in warning for warning in warnings)
    assert any("missing values" in warning for warning in warnings)
    assert any("missing daily periods" in warning for warning in warnings)
    assert any("Missing-data policy dropped columns" in warning for warning in warnings)
    assert any("Sanitized column headers" in warning for warning in warnings)
    assert meta["symbols"] == ["=Bad", "Index"]
    assert meta["date_range"] == ("2020-01-01", "2020-01-05")


def test_read_binary_payload_rejects_unsupported_object():
    class NotReadable:
        pass

    with pytest.raises(TypeError):
        load_and_validate_file(NotReadable())
