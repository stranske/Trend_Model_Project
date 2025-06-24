import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from trend_analysis.data import load_csv


def test_load_csv_success(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "A": [1, 2]}).to_csv(csv_path, index=False)
    df = load_csv(str(csv_path))
    assert df is not None and len(df) == 2


def test_load_csv_missing_file(tmp_path):
    assert load_csv(str(tmp_path / "missing.csv")) is None


def test_load_csv_no_date(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"X": [1]}).to_csv(csv_path, index=False)
    assert load_csv(str(csv_path)) is None


def test_load_csv_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    assert load_csv(str(csv_path)) is None


def test_load_csv_parse_error(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("not,a,csv\n1,2")
    assert load_csv(str(csv_path)) is None


def test_load_csv_null_dates(tmp_path):
    csv_path = tmp_path / "null.csv"
    csv_path.write_text("Date,A\n,1\n2020-02-01,2")
    df = load_csv(str(csv_path))
    assert df is not None
