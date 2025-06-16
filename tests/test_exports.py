import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from exports import export_data


def test_export_data(tmp_path):
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"B": [3, 4]})
    data = {"sheet1": df1, "sheet2": df2}
    out = tmp_path / "report"
    export_data(data, str(out), formats=["xlsx", "csv", "json"])

    assert (tmp_path / "report.xlsx").exists()
    assert (tmp_path / "report_sheet1.csv").exists()
    assert (tmp_path / "report_sheet2.csv").exists()
    assert (tmp_path / "report_sheet1.json").exists()
    assert (tmp_path / "report_sheet2.json").exists()

    read = pd.read_csv(tmp_path / "report_sheet1.csv")
    pd.testing.assert_frame_equal(read, df1)
