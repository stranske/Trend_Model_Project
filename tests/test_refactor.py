import sys
import pathlib
import pandas as pd

# Ensure repository root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_utils import load_csv


def test_load_csv_missing_date(tmp_path):
    data = pd.DataFrame({'A': [1, 2, 3]})
    csv_path = tmp_path / "nodate.csv"
    data.to_csv(csv_path, index=False)

    result = load_csv(str(csv_path))
    assert result is None
