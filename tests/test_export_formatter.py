from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.export import (
    FORMATTERS_EXCEL,
    _maybe_remove_openpyxl_default_sheet,
    _normalise_color,
    export_to_excel,
    format_summary_text,
    make_period_formatter,
    make_summary_formatter,
)

pytestmark = pytest.mark.cosmetic


pytestmark = pytest.mark.cosmetic


@pytest.fixture
def formatters_excel_registry():
    """Yield with a clean FORMATTERS_EXCEL registry and restore afterward."""
    original = FORMATTERS_EXCEL.copy()
    FORMATTERS_EXCEL.clear()
    try:
        yield
    finally:
        FORMATTERS_EXCEL.clear()
        FORMATTERS_EXCEL.update(original)


class DummyWS:
    def __init__(self):
        self.rows = []
        self.cells = []
        self.columns = []
        self.frozen = None
        self.autofilter_args = None

    def write_row(self, row, col, data, fmt=None):
        self.rows.append((row, col, list(data), fmt))

    def write(self, row, col, data, fmt=None):
        self.cells.append((row, col, data, fmt))

    def set_column(self, c1, c2, width):
        self.columns.append((c1, c2, width))

    def freeze_panes(self, row, col):
        self.frozen = (row, col)

    def autofilter(self, fr, fc, lr, lc):
        self.autofilter_args = (fr, fc, lr, lc)


class DummyWB:
    def __init__(self):
        self.formats = []

    def add_format(self, spec):
        self.formats.append(spec)
        return spec


class DummyOpenpyxlSheet:
    def __init__(self, title: str = "Sheet", value: object | None = None) -> None:
        self.title = title
        self._value = value

    def cell(self, row: int, column: int) -> SimpleNamespace:  # noqa: ARG002
        return SimpleNamespace(value=self._value)


class DummyOpenpyxlBook:
    def __init__(self, title: str = "Sheet", value: object | None = None) -> None:
        self.worksheets: list[DummyOpenpyxlSheet] = [
            DummyOpenpyxlSheet(title=title, value=value)
        ]
        self._removed: list[DummyOpenpyxlSheet] = []

    def remove(self, sheet: DummyOpenpyxlSheet) -> None:
        self._removed.append(sheet)
        self.worksheets = [ws for ws in self.worksheets if ws is not sheet]


def test_normalise_color_handles_variants():
    assert _normalise_color(" red ") == "FFFF0000"
    assert _normalise_color("#1a2b3c") == "FF1A2B3C"
    assert _normalise_color("ff89abcd") == "FF89ABCD"
    assert _normalise_color("  ") is None
    assert _normalise_color(123) is None
    assert _normalise_color("not-a-colour") is None


def test_maybe_remove_openpyxl_default_sheet_prunes_blank_sheet():
    book = DummyOpenpyxlBook()
    removed_title = _maybe_remove_openpyxl_default_sheet(book)
    assert removed_title == "Sheet"
    assert book._removed and book._removed[0].title == "Sheet"
    assert book.worksheets == []


def test_maybe_remove_openpyxl_default_sheet_ignores_named_or_populated():
    named_book = DummyOpenpyxlBook(title="Report")
    assert _maybe_remove_openpyxl_default_sheet(named_book) is None
    assert named_book.worksheets

    populated_book = DummyOpenpyxlBook(title="Sheet", value="data")
    assert _maybe_remove_openpyxl_default_sheet(populated_book) is None
    assert populated_book.worksheets


def test_make_summary_formatter_registers_and_runs(formatters_excel_registry):
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    assert "summary" in FORMATTERS_EXCEL
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.rows[0][2][0] == "Vol-Adj Trend Analysis"
    meta_rows = [row for row in ws.rows if row[0] == 3]
    assert meta_rows and meta_rows[0][2][0].startswith("Frequency:")


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_make_summary_formatter_writes_manager_contrib(
    formatters_excel_registry, as_dataframe
):
    """Manager contribution payloads should render a dedicated section."""

    stats = (0.01, 0.02, 0.5, 0.4, 0.3, -0.1)
    rows = [
        {"Manager": "FundA", "Years": 1.5, "OOS CAGR": 0.04, "Contribution Share": 0.6},
        {"Manager": "FundB", "Years": 2.0, "OOS CAGR": 0.03, "Contribution Share": 0.4},
    ]
    contrib_payload = pd.DataFrame(rows) if as_dataframe else rows
    res = {
        "in_ew_stats": stats,
        "out_ew_stats": stats,
        "in_user_stats": stats,
        "out_user_stats": stats,
        "in_sample_stats": {"FundA": stats},
        "out_sample_stats": {"FundA": stats},
        "fund_weights": {"FundA": 0.5},
        "manager_contrib": contrib_payload,
        "manager_changes": [],
    }

    fmt = make_summary_formatter(res, "2020-01", "2020-12", "2021-01", "2021-12")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)

    headers = [entry[2] for entry in ws.rows]
    assert any(
        row == ["Manager Participation & Contribution"] for row in headers
    ), "Expected contribution section header"
    contrib_headers = [
        row
        for row in headers
        if row == ["Manager", "Years", "OOS CAGR", "Contribution Share"]
    ]
    assert contrib_headers, "Expected contribution column headers"

    managers_written = [
        cell[2] for cell in ws.cells if cell[1] == 0 and cell[2] in {"FundA", "FundB"}
    ]
    assert set(managers_written) == {"FundA", "FundB"}


def test_format_summary_text_basic():
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    text = format_summary_text(res, "a", "b", "c", "d")
    assert "Vol-Adj Trend Analysis" in text
    assert "fund" in text


def test_format_summary_text_includes_regime_breakdown():
    regime_table = pd.DataFrame(
        {
            ("User", "Risk-On"): [0.12, 1.2, -0.18, 0.58, 24],
            ("User", "Risk-Off"): [0.03, 0.3, -0.25, 0.42, 8],
        },
        index=["CAGR", "Sharpe", "Max Drawdown", "Hit Rate", "Observations"],
    )
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
        "performance_by_regime": regime_table,
        "regime_summary": "Risk-On windows delivered 12% CAGR versus 3% in risk-off.",
        "regime_notes": ["Risk-Off windows have limited observations."],
    }
    text = format_summary_text(res, "a", "b", "c", "d")

    assert "Performance by regime" in text
    assert "Risk-On" in text
    assert "Risk-Off" in text
    assert "Regime insight:" in text
    assert "Regime notes:" in text


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
        "in_ew_stats": (0, 0, 0, 0, 0, 0),
        "out_ew_stats": (0, 0, 0, 0, 0, 0),
        "in_user_stats": (0, 0, 0, 0, 0, 0),
        "out_user_stats": (0, 0, 0, 0, 0, 0),
        "in_sample_stats": {},
        "out_sample_stats": {},
        "fund_weights": {},
    }

    sheet_formatter = make_summary_formatter(res, "a", "b", "c", "d")
    out = tmp_path / "out.xlsx"
    export_to_excel({"summary": df}, str(out), formatter=sheet_formatter)
    assert out.exists()


def test_export_to_excel_without_xlsxwriter(
    formatters_excel_registry, monkeypatch, tmp_path
):
    df = pd.DataFrame({"A": [1]})
    called: list[str] = []

    def sheet_formatter(ws, wb):
        called.append(ws.name)
        ws.write_row(0, 0, df.columns)
        for idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
            ws.write_row(idx, 0, row)

    original_excel_writer = pd.ExcelWriter

    def fake_excel_writer(path, engine=None, **kwargs):
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("xlsxwriter missing")
        # Force openpyxl so the workbook lacks ``add_worksheet`` / ``add_format`` APIs.
        return original_excel_writer(path, engine="openpyxl", **kwargs)

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    out = tmp_path / "fallback.xlsx"
    export_to_excel({"summary": df}, str(out), formatter=sheet_formatter)

    assert out.exists()
    assert called == ["summary"]
    read = pd.read_excel(out, sheet_name="summary")
    pd.testing.assert_frame_equal(read, df)


def test_format_summary_text_no_index_stats():
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    text = format_summary_text(res, "a", "b", "c", "d")
    assert "fund" in text


def test_make_summary_formatter_handles_nan(tmp_path):
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, float("nan")),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, float("nan"), 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.rows[0][2][0] == "Vol-Adj Trend Analysis"


def test_make_summary_formatter_with_benchmarks(formatters_excel_registry):
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 1.0},
        "benchmark_ir": {"spx": {"fund": 0.1, "equal_weight": 0.2, "user_weight": 0.3}},
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    header = next(r for r in ws.rows if r[0] == 4)[2]
    assert "OS IR spx" in header


def test_make_summary_formatter_contract():
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 1.0},
    }
    fmt = make_summary_formatter(res, "a", "b", "c", "d")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.frozen == (5, 0)
    assert ws.autofilter_args[0] == 4
    assert ws.columns[0][2] == len("Name") + 2


def test_make_period_formatter_registers_and_runs(formatters_excel_registry):
    res = {
        "in_ew_stats": (1, 1, 1, 1, 1, 1),
        "out_ew_stats": (2, 2, 2, 2, 2, 2),
        "in_user_stats": (3, 3, 3, 3, 3, 3),
        "out_user_stats": (4, 4, 4, 4, 4, 4),
        "in_sample_stats": {"fund": (5, 5, 5, 5, 5, 5)},
        "out_sample_stats": {"fund": (6, 6, 6, 6, 6, 6)},
        "fund_weights": {"fund": 0.5},
    }
    in_start = "2020-01-01"
    in_end = "2020-06-30"
    out_start = "2020-07-01"
    out_end = "2020-12-31"
    fmt = make_period_formatter("period_1", res, in_start, in_end, out_start, out_end)
    assert "period_1" in FORMATTERS_EXCEL
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)
    assert ws.rows[0][2][0] == "Vol-Adj Trend Analysis"


def _build_base_result():
    stats_a = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    stats_b = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    return {
        "in_ew_stats": stats_a,
        "out_ew_stats": stats_b,
        "in_user_stats": stats_a,
        "out_user_stats": stats_b,
        "in_sample_stats": {"Fund A": stats_a},
        "out_sample_stats": {"Fund A": stats_b},
        "fund_weights": {"Fund A": None},
    }


def test_make_summary_formatter_optional_sections(formatters_excel_registry):
    res = _build_base_result()
    res["benchmark_ir"] = {
        "bench": {
            "Fund A": pd.NA,
            "equal_weight": None,
            "user_weight": pd.NA,
        }
    }
    res["manager_changes"] = [
        {"Period": "2020", "action": "add", "manager": "Fund A", "detail": "init"},
        {"action": "remove", "manager": "Fund B"},
    ]
    res["manager_contrib"] = pd.DataFrame(
        [
            {
                "Manager": "Fund A",
                "Years": 2,
                "OOS CAGR": 0.05,
                "Contribution Share": 0.25,
            }
        ]
    )

    fmt = make_summary_formatter(res, "2020-01", "2020-06", "2020-07", "2020-12")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)

    headers = [row for row in ws.rows if row[0] >= 0]
    assert any("Manager Changes" in row[2] for row in headers)
    assert any("Manager Participation & Contribution" in row[2] for row in headers)
    # The optional sections should also write out manager rows
    assert any(cell[2] == "Fund A" for cell in ws.cells)


def test_make_summary_formatter_manager_contrib_list(formatters_excel_registry):
    res = _build_base_result()
    res["benchmark_ir"] = {
        "bench": {"Fund A": 0.1, "equal_weight": 0.2, "user_weight": 0.3}
    }
    res["manager_contrib"] = [
        {"Manager": "Fund B", "Years": 3, "OOS CAGR": 0.02, "Contribution Share": 0.1}
    ]

    fmt = make_summary_formatter(res, "2020-01", "2020-06", "2020-07", "2020-12")
    ws = DummyWS()
    wb = DummyWB()
    fmt(ws, wb)

    contrib_rows = [row for row in ws.rows if row[2] and row[2][0] == "Manager"]
    assert contrib_rows, "Expected contribution header row to be written"
    # Ensure list-based records were written to worksheet
    assert any(cell[2] == "Fund B" for cell in ws.cells)


def test_format_summary_text_formats_ints_and_nones():
    res = _build_base_result()
    res["fund_weights"] = {"Fund A": 1}
    res["benchmark_ir"] = {
        "bench": {"Fund A": pd.NA, "equal_weight": None, "user_weight": 0.1}
    }

    text = format_summary_text(res, "2020-01", "2020-06", "2020-07", "2020-12")
    # Weight of 1 -> 100% formatted with two decimals
    assert "100.00" in text
