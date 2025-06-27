import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trend_analysis.export import (
    export_data,
    export_to_excel,
    register_formatter_excel,
    reset_formatters_excel,
    FORMATTERS_EXCEL,
)


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

    read = pd.read_csv(tmp_path / "report_sheet1.csv", index_col=0)
    pd.testing.assert_frame_equal(read, df1)


def test_export_data_excel_alias(tmp_path):
    df = pd.DataFrame({"A": [1]})
    data = {"sheet": df}
    out = tmp_path / "alias"
    export_data(data, str(out), formats=["excel"])
    assert (tmp_path / "alias.xlsx").exists()


def test_export_to_excel_formatters(tmp_path):
    reset_formatters_excel()

    calls: list[str] = []

    @register_formatter_excel("sheet1")
    def fmt1(ws, wb):
        calls.append("fmt1")

    def default(ws, wb):
        calls.append("default")

    df1 = pd.DataFrame({"A": [1]})
    df2 = pd.DataFrame({"B": [2]})
    export_to_excel(
        {"sheet1": df1, "sheet2": df2},
        str(tmp_path / "out.xlsx"),
        default_sheet_formatter=default,
    )

    assert calls == ["fmt1", "default"]


def test_reset_formatters_excel():
    reset_formatters_excel()

    @register_formatter_excel("foo")
    def fmt(ws, wb):
        pass

    assert "foo" in FORMATTERS_EXCEL
    reset_formatters_excel()
    assert FORMATTERS_EXCEL == {}
