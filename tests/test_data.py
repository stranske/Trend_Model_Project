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


