import pandas as pd

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
