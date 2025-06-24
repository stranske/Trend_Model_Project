import logging
import pandas as pd
import pytest
from trend_analysis.data import load_csv


def test_load_csv_success(tmp_path, caplog):
    p = tmp_path / "ok.csv"
    df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=2), "A": [1, 2]})
    df.to_csv(p, index=False)
    with caplog.at_level(logging.WARNING):
        loaded = load_csv(str(p))
    pd.testing.assert_frame_equal(loaded, df)


def test_load_csv_missing_file(tmp_path):
    assert load_csv(str(tmp_path / "none.csv")) is None


def test_load_csv_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert load_csv(str(p)) is None


def test_load_csv_parser_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text('A,B\n1,"2')
    assert load_csv(str(p)) is None


def test_load_csv_no_date_column(tmp_path):
    p = tmp_path / "nodate.csv"
    pd.DataFrame({"A": [1]}).to_csv(p, index=False)
    assert load_csv(str(p)) is None


def test_load_csv_missing_date_after_read(monkeypatch, tmp_path):
    def fake_read_csv(*args, **kwargs):
        return pd.DataFrame({"A": [1]})
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    assert load_csv("irrelevant.csv") is None


def test_load_csv_null_dates(tmp_path, caplog):
    p = tmp_path / "null.csv"
    df = pd.DataFrame({"Date": [pd.NaT, pd.Timestamp("2020-01-01")], "A": [1, 2]})
    df.to_csv(p, index=False)
    with caplog.at_level(logging.WARNING):
        loaded = load_csv(str(p))
    assert "Null values" in caplog.text
    assert loaded is not None
