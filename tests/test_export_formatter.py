import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trend_analysis.export import (
    make_summary_formatter,
    export_to_excel,
    FORMATTERS_EXCEL,
    format_summary_text,
)


class DummyWS:
    def __init__(self):
        self.rows = []
        self.cells = []

    def write_row(self, row, col, data, fmt=None):
        self.rows.append((row, col, list(data), fmt))

    def write(self, row, col, data, fmt=None):
        self.cells.append((row, col, data, fmt))


class DummyWB:
    def __init__(self):
        self.formats = []

    def add_format(self, spec):
        self.formats.append(spec)
        return spec


def test_make_summary_formatter_registers_and_runs():
    FORMATTERS_EXCEL.clear()
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
        "index_stats": {
            "idx": {"in_sample": (7, 7, 7, 7, 7), "out_sample": (8, 8, 8, 8, 8)}
        },
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    assert "summary" in FORMATTERS_EXCEL
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.rows[0][2][0] == "Vol-Adj Trend Analysis"


def test_format_summary_text_basic():
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
        "index_stats": {},
    }
    text = format_summary_text(res, "a", "b", "c", "d")
    assert "Vol-Adj Trend Analysis" in text
    assert "fund" in text


def test_export_to_excel_invokes_formatter(tmp_path):
    df = pd.DataFrame({"A": [1]})
    called = []

    def formatter(frame: pd.DataFrame) -> pd.DataFrame:
        called.append(True)
        out = frame.copy()
        out["B"] = 2
        return out

    out = tmp_path / "data.xlsx"
    export_to_excel({"sheet": df}, str(out), formatter=formatter)

    assert called == [True]
    read = pd.read_excel(out, sheet_name="sheet")
    assert "B" in read.columns
    assert read.loc[0, "B"] == 2


def test_export_to_excel_backward_compat_sheet_formatter(tmp_path):
    df = pd.DataFrame({"A": [1]})
    res = {
        "in_ew_stats": (0, 0, 0, 0, 0),
        "out_ew_stats": (0, 0, 0, 0, 0),
        "in_user_stats": (0, 0, 0, 0, 0),
        "out_user_stats": (0, 0, 0, 0, 0),
        "in_sample_stats": {},
        "out_sample_stats": {},
        "fund_weights": {},
        "index_stats": {},
    }

    sheet_formatter = make_summary_formatter(res, "a", "b", "c", "d")
    out = tmp_path / "out.xlsx"
    export_to_excel({"summary": df}, str(out), formatter=sheet_formatter)
    assert out.exists()

def test_format_summary_text_no_index_stats():
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    text = format_summary_text(res, "a", "b", "c", "d")
    assert "fund" in text


def test_make_summary_formatter_handles_nan(tmp_path):
    res = {
        "in_ew_stats": (1, 1, 1, 1, float("nan")),
        "out_ew_stats": (2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, float("nan"), 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
        "index_stats": {},
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.rows[0][2][0] == "Vol-Adj Trend Analysis"
