

import pandas as pd
import pytest
from types import SimpleNamespace
from trend_analysis import data as data_mod


def test_load_csv_ok(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,A\n2020-01-01,1")
    df = data_mod.load_csv(str(f))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Date", "A"]


def test_load_csv_missing(tmp_path):
    assert data_mod.load_csv(str(tmp_path / "none.csv")) is None


def test_load_csv_empty(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    assert data_mod.load_csv(str(f)) is None


def test_load_csv_parser_error(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text('Date,A\n"2020-01-01,1')
    assert data_mod.load_csv(str(f)) is None


def test_load_csv_value_error(tmp_path):
    f = tmp_path / "nodate.csv"
    f.write_text("A,B\n1,2")
    assert data_mod.load_csv(str(f)) is None


def test_load_csv_missing_date_column(monkeypatch, tmp_path):
    f = tmp_path / "dummy.csv"
    f.write_text("X\n1")

    def fake_read(*args, **kwargs):
        return pd.DataFrame({"X": [1]})

    monkeypatch.setattr(data_mod.pd, "read_csv", fake_read)
    assert data_mod.load_csv(str(f)) is None


def test_load_csv_null_dates(tmp_path):
    f = tmp_path / "null.csv"
    f.write_text("Date,A\n,1\n2020-01-01,2")
    df = data_mod.load_csv(str(f))
    assert df is not None


def test_load_csv_malformed_date_strings(tmp_path, caplog):
    f = tmp_path / "malformed_dates.csv"
    f.write_text("Date,A\nbad,1\n01/01/20,2")
    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(f))
    assert result is None
    assert "malformed date" in caplog.text


def test_load_csv_malformed_dates_preview_tail(tmp_path, caplog):
    f = tmp_path / "malformed_many_dates.csv"
    bad_rows = "\n".join([f"bad{i},1" for i in range(7)])
    f.write_text(f"Date,A\n{bad_rows}")

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(f))

    assert result is None
    assert "7 malformed date(s)" in caplog.text
    assert "..." in caplog.text


def test_load_csv_permission_error_during_stat(monkeypatch, tmp_path, caplog):
    f = tmp_path / "stat_permission.csv"
    f.write_text("Date,A\n01/01/20,1")

    original_stat = data_mod.Path.stat

    def fake_stat(self, *args, **kwargs):
        if self == data_mod.Path(str(f)):
            raise PermissionError("stat denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(data_mod.Path, "stat", fake_stat)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(f))
    assert result is None
    assert "Permission denied" in caplog.text


def test_load_csv_not_readable_mode(monkeypatch, tmp_path, caplog):
    """Files without read permissions should trigger a logged error."""

    f = tmp_path / "noread.csv"
    f.write_text("Date,A\n01/01/20,1")

    monkeypatch.setattr(
        data_mod.Path, "stat", lambda self, **_: SimpleNamespace(st_mode=0)
    )

    with caplog.at_level("ERROR"):
        assert data_mod.load_csv(str(f)) is None

    assert "Permission denied" in caplog.text


def test_load_csv_null_dates_all_filtered(tmp_path, monkeypatch):
    f = tmp_path / "null_only.csv"
    f.write_text("Date,A\n,1\n,2")

    original_to_datetime = data_mod.pd.to_datetime

    def fake_to_datetime(values, *args, **kwargs):
        if kwargs.get("errors") == "coerce":
            return original_to_datetime(values, *args, **kwargs)
        raise ValueError("force fallback")

    monkeypatch.setattr(data_mod.pd, "to_datetime", fake_to_datetime)
    result = data_mod.load_csv(str(f))
    assert result is None


def test_load_csv_all_null_dates_logs_error(tmp_path, monkeypatch, caplog):
    f = tmp_path / "null_only.csv"
    f.write_text("Date,A\n,1\n,2")

    original_to_datetime = data_mod.pd.to_datetime

    def fake_to_datetime(values, *args, **kwargs):
        if kwargs.get("format") == "%m/%d/%y":
            raise ValueError("force fallback")
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(data_mod.pd, "to_datetime", fake_to_datetime)

    with caplog.at_level("ERROR"):
        result = data_mod.load_csv(str(f))

    assert result is None
    assert "No valid date rows remaining" in caplog.text


def test_load_csv_numeric_and_percent_coercion(tmp_path):
    f = tmp_path / "coerce_numeric.csv"
    f.write_text(
        """Date,Value,Percent,Neg
01/01/20,"1,234",50%,(100)
01/02/20,"2,468",75%,(200)
"""
    )
    df = data_mod.load_csv(str(f))
    assert df is not None
    assert df["Value"].tolist() == [1234.0, 2468.0]
    assert df["Percent"].tolist() == [0.5, 0.75]
    assert df["Neg"].tolist() == [-100.0, -200.0]


def test_load_csv_warns_on_null_dates_without_format_error(tmp_path, caplog):
    f = tmp_path / "null_warning.csv"
    f.write_text("Date,A\n,1\n01/01/20,2")
    with caplog.at_level("WARNING"):
        df = data_mod.load_csv(str(f))
    assert df is not None
    assert df["Date"].isnull().sum() == 1
    assert "Null values found" in caplog.text


def test_load_csv_null_dates_preview_tail(monkeypatch, tmp_path, caplog):
    f = tmp_path / "null_many_dates.csv"
    rows = [",1" for _ in range(7)] + ["01/01/20,2"]
    f.write_text("Date,A\n" + "\n".join(rows))

    original_to_datetime = data_mod.pd.to_datetime

    def fake_to_datetime(values, *args, **kwargs):
        if kwargs.get("format") == "%m/%d/%y":
            raise ValueError("force fallback")
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(data_mod.pd, "to_datetime", fake_to_datetime)

    with caplog.at_level("WARNING"):
        df = data_mod.load_csv(str(f))

    assert df is not None
    assert df.shape[0] == 1
    assert df["A"].tolist() == [2]
    assert "Removing these rows" in caplog.text
    assert "..." in caplog.text


def test_load_csv_fallback_parses_clean_data(monkeypatch, tmp_path):
    f = tmp_path / "fallback.csv"
    f.write_text("Date,A\n2020-01-01,1\n2020-02-01,2")

    original_to_datetime = data_mod.pd.to_datetime

    def fake_to_datetime(values, *args, **kwargs):
        if kwargs.get("format") == "%m/%d/%y":
            raise ValueError("force fallback")
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(data_mod.pd, "to_datetime", fake_to_datetime)

    df = data_mod.load_csv(str(f))
    assert df is not None
    assert df["Date"].dt.month.tolist() == [1, 2]


def test_load_csv_preserves_non_numeric_strings(monkeypatch, tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text("Date,Label\n01/01/20,alpha\n01/02/20,beta")

    original_to_numeric = data_mod.pd.to_numeric

    def fake_to_numeric(values, *args, **kwargs):
        return pd.Series(list(values), index=values.index, dtype=object)

    monkeypatch.setattr(data_mod.pd, "to_numeric", fake_to_numeric)

    df = data_mod.load_csv(str(f))

    assert df is not None
    assert df["Label"].tolist() == ["alpha", "beta"]

    # Restore to ensure other tests continue with the original implementation
    monkeypatch.setattr(data_mod.pd, "to_numeric", original_to_numeric)


def test_identify_risk_free_fund_basic():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A": [0.02, 0.01, 0.03],
            "B": [0.01, 0.01, 0.01],
        }
    )
    assert data_mod.identify_risk_free_fund(df) == "B"


def test_identify_risk_free_fund_no_numeric():
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": ["x"]})
    assert data_mod.identify_risk_free_fund(df) is None


def test_ensure_datetime_coerces():
    df = pd.DataFrame({"Date": ["2020-01-01"], "A": [1]})
    out = data_mod.ensure_datetime(df)
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_ensure_datetime_missing_column():
    df = pd.DataFrame({"X": [1]})
    out = data_mod.ensure_datetime(df)
    assert "X" in out.columns and "Date" not in out.columns


def test_ensure_datetime_raises_on_malformed():
    df = pd.DataFrame({"Date": ["not-a-date", "01/01/20"]})
    with pytest.raises(ValueError, match="Malformed dates"):
        data_mod.ensure_datetime(df)


def test_load_csv_permission_error(tmp_path):
    """Test permission error handling."""
    f = tmp_path / "restricted.csv"
    f.write_text("Date,A\n2020-01-01,1")

    # Remove read permissions
    f.chmod(0o000)

    try:
        result = data_mod.load_csv(str(f))
        assert result is None
    finally:
        # Restore permissions for cleanup
        f.chmod(0o644)


def test_load_csv_directory_error(tmp_path):
    """Test directory path error handling."""
    # Try to load a directory as a CSV file
    result = data_mod.load_csv(str(tmp_path))
    assert result is None
