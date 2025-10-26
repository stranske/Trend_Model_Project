import sys
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd

import trend_analysis.export as export


class DummyColumnDim:
    def __init__(self) -> None:
        self.width: float | None = None


class DummyFont:
    def __init__(self) -> None:
        self.color: str | None = None

    def copy(self, **kwargs) -> "DummyFont":
        new = DummyFont()
        new.color = kwargs.get("color")
        return new


class DummyWorksheet:
    def __init__(self, title: str = "Sheet") -> None:
        self.title = title
        self._cells: dict[tuple[int, int], SimpleNamespace] = {}
        self.column_dimensions: defaultdict[str, DummyColumnDim] = defaultdict(
            DummyColumnDim
        )
        self.freeze_panes = None
        self.auto_filter = SimpleNamespace(ref="")

    def cell(
        self, row: int, column: int, value: object | None = None
    ) -> SimpleNamespace:
        key = (row, column)
        cell = self._cells.setdefault(
            key,
            SimpleNamespace(value=None, number_format=None, font=DummyFont()),
        )
        if value is not None:
            cell.value = value
        return cell


class DummyWorkbook:
    def __init__(self) -> None:
        self.worksheets: list[DummyWorksheet] = [DummyWorksheet()]
        self.foo = "bar"

    def remove(self, sheet: DummyWorksheet) -> None:
        if sheet in self.worksheets:
            self.worksheets.remove(sheet)

    def create_sheet(self, title: str) -> DummyWorksheet:
        ws = DummyWorksheet(title)
        self.worksheets.append(ws)
        return ws

    def __getitem__(self, key: str) -> DummyWorksheet:
        raise KeyError(key)


def test_openpyxl_adapter_wraps_core_methods(monkeypatch):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    wb = DummyWorkbook()
    adapter = export._OpenpyxlWorkbookAdapter(wb)
    ws = adapter.add_worksheet("Report")
    ws.write(0, 0, "value")
    ws.write_row(1, 0, [1, 2])
    ws.set_column(0, 1, 12)
    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, 1, 1)

    native = wb.worksheets[-1]
    assert native.title == "Report"
    assert native.column_dimensions["A"].width == 12
    assert native.auto_filter.ref == "A1:B2"

    adapter.rename_last_sheet("Final")
    assert wb.worksheets[-1].title == "Final"


def test_export_to_excel_uses_adapter_when_xlsxwriter_missing(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class DummyWriter:
        def __init__(self) -> None:
            self.book = DummyWorkbook()
            self.sheets: dict[str, object] = {}

        def __enter__(self) -> "DummyWriter":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

    created: list[DummyWriter] = []

    def fake_excel_writer(path, engine=None):  # noqa: ARG001
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("no xlsxwriter")
        writer = DummyWriter()
        created.append(writer)
        return writer

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    def fake_to_excel(self, writer, sheet_name, index=False, **_: object) -> None:  # noqa: ARG002
        writer.sheets[sheet_name] = object()
        writer.book.worksheets.append(DummyWorksheet("Sheet"))

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    export.reset_formatters_excel()
    df = pd.DataFrame({"x": [1]})
    out = tmp_path / "report.xlsx"
    export.export_to_excel({"summary": df}, out)

    assert export.FORMATTERS_EXCEL == {}
    assert created and created[0].book.worksheets[-1].title == "summary"


def test_openpyxl_proxy_apply_format_sets_styles(monkeypatch):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    ws = DummyWorksheet()
    proxy = export._OpenpyxlWorksheetProxy(ws)

    proxy.write(
        0,
        0,
        "value",
        {"num_format": "0.00", "font_color": "#123abc"},
    )

    cell = ws.cell(1, 1)
    assert cell.number_format == "0.00"
    # Normalised hex should have an FF alpha prefix
    assert cell.font.color == "FF123ABC"


def test_openpyxl_proxy_skips_formatting_when_letter_helper_missing(monkeypatch):
    ws = DummyWorksheet()
    proxy = export._OpenpyxlWorksheetProxy(ws)

    monkeypatch.setattr(export, "get_column_letter", None)

    proxy.set_column(0, 1, 10)
    proxy.autofilter(0, 0, 0, 0)

    assert len(ws.column_dimensions) == 0
    assert ws.auto_filter.ref == ""


def test_openpyxl_proxy_ignores_invalid_font_color():
    ws = DummyWorksheet()
    proxy = export._OpenpyxlWorksheetProxy(ws)

    proxy.write(0, 0, "value", {"font_color": "invalid"})

    cell = ws.cell(1, 1)
    assert cell.font.color is None


def test_openpyxl_workbook_proxy_removes_default_sheet(monkeypatch):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    writer = SimpleNamespace(book=DummyWorkbook(), sheets={"Sheet": object()})
    proxy = export._OpenpyxlWorkbookProxy(writer)

    ws_proxy = proxy.add_worksheet("Report")

    assert "Sheet" not in writer.sheets
    assert ws_proxy.name == "Report"


def test_export_to_excel_cleans_up_openpyxl_defaults(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class FakeOpenpyxlWorkbook(DummyWorkbook):
        __module__ = "openpyxl.workbook"

    class TrackingDict(dict[str, object]):
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN001
            super().__init__(*args, **kwargs)
            self.pops: list[str] = []

        def pop(self, key, default=None):  # noqa: D401, ANN001
            self.pops.append(key)
            return super().pop(key, default)

    class DummyWriter:
        def __init__(self) -> None:
            self.book = FakeOpenpyxlWorkbook()
            self.sheets: TrackingDict = TrackingDict({"Sheet": object()})
            self.engine = "openpyxl"

        def __enter__(self) -> "DummyWriter":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

    created: list[DummyWriter] = []
    removals = {"count": 0}

    def fake_excel_writer(path, engine=None):  # noqa: ARG001
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("no xlsxwriter")
        writer = DummyWriter()
        created.append(writer)
        return writer

    def fake_remove_default(book):  # noqa: ANN001
        removals["count"] += 1
        return "Sheet"

    def fake_to_excel(self, writer, sheet_name, index=False, **_):  # noqa: ARG002
        ws = DummyWorksheet("Sheet1")
        writer.book.worksheets.append(ws)
        writer.sheets["Sheet1"] = ws

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)
    monkeypatch.setattr(
        export, "_maybe_remove_openpyxl_default_sheet", fake_remove_default
    )
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    export.reset_formatters_excel()
    df = pd.DataFrame({"x": [1.0]})
    out = tmp_path / "cleanup.xlsx"
    export.export_to_excel({"report": df}, out)

    assert created, "Expected fallback writer to be created"
    writer = created[0]
    assert "Sheet" not in writer.sheets
    assert "Sheet" in writer.sheets.pops
    ws = writer.book.worksheets[-1]
    assert ws.title == "report"
    assert writer.sheets.get("report") is ws
    assert removals["count"] >= 2


def test_export_to_excel_handles_missing_sheet_lookup(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    rename_calls: list[tuple[str, str | None]] = []

    class TrackingAdapter:
        def __init__(self, book):  # noqa: ANN001
            self.book = book
            export._maybe_remove_openpyxl_default_sheet(book)

        def rename_last_sheet(self, name: str) -> None:
            last_title = (
                self.book.worksheets[-1].title if self.book.worksheets else None
            )
            rename_calls.append((name, last_title))
            if self.book.worksheets:
                self.book.worksheets[-1].title = name

    monkeypatch.setattr(export, "_OpenpyxlWorkbookAdapter", TrackingAdapter)

    class NoLookupWorkbook(DummyWorkbook):
        __module__ = "openpyxl.workbook"

        def __getitem__(self, key: str) -> DummyWorksheet:  # noqa: D401
            raise KeyError(key)

    class DummyWriter:
        def __init__(self) -> None:
            self.book = NoLookupWorkbook()
            self.sheets: dict[str, object] = {"Sheet": object()}
            self.engine = "openpyxl"

        def __enter__(self) -> "DummyWriter":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    writers: list[DummyWriter] = []

    def fake_excel_writer(path, engine=None):  # noqa: ANN001, ARG001
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("no xlsxwriter")
        writer = DummyWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    def fake_to_excel(self, writer, sheet_name, index=False, **_):  # noqa: ARG002
        ws = DummyWorksheet("Temp")
        writer.book.worksheets.append(ws)

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    export.reset_formatters_excel()
    df = pd.DataFrame({"value": [1.0]})
    out = tmp_path / "lookup.xlsx"
    export.export_to_excel({"summary": df}, out)

    assert writers, "Expected fallback writer"
    writer = writers[0]
    assert rename_calls and rename_calls[-1][0] == "summary"
    final_sheet = writer.book.worksheets[-1]
    assert final_sheet.title == "summary"
    assert writer.sheets["summary"] is final_sheet


def test_export_to_excel_populates_proxy_with_renamed_sheets(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class DummyWriter:
        def __init__(self) -> None:
            self.book = DummyWorkbook()
            self.sheets: dict[str, object] = {"Sheet": object()}
            self.engine = "openpyxl"

        def __enter__(self) -> "DummyWriter":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

    created: list[DummyWriter] = []

    def fake_excel_writer(path, engine=None):  # noqa: ARG001
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("no xlsxwriter")
        writer = DummyWriter()
        created.append(writer)
        return writer

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    def fake_to_excel(self, writer, sheet_name, index=False, **_):  # noqa: ARG002
        ws = DummyWorksheet("Temp")
        writer.book.worksheets.append(ws)
        writer.sheets[sheet_name] = ws

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    export.reset_formatters_excel()
    captured: dict[str, object] = {}

    def fake_formatter(ws, wb):  # noqa: ANN001
        captured["formatter_called"] = (ws, wb)

    export.register_formatter_excel("metrics")(fake_formatter)

    df = pd.DataFrame({"x": [1.0]})
    out = tmp_path / "report.xlsx"

    export.export_to_excel({"metrics": df, "summary": df}, out)

    assert created, "Expected ExcelWriter to be instantiated"
    assert captured["formatter_called"][0].name == "metrics"
    metrics_ws = created[0].sheets.get("metrics")
    summary_ws = created[0].sheets.get("summary")
    assert isinstance(metrics_ws, DummyWorksheet)
    assert metrics_ws.title == "metrics"
    assert isinstance(summary_ws, DummyWorksheet)
    assert summary_ws.title == "summary"
    export.reset_formatters_excel()


def test_export_to_excel_strips_temp_sheet_key(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class DummyWriter:
        def __init__(self) -> None:
            self.book = DummyWorkbook()
            self.sheets: dict[str, object] = {"Sheet": object()}
            self.engine = "openpyxl"

        def __enter__(self) -> "DummyWriter":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

    created: list[DummyWriter] = []

    def fake_excel_writer(path, engine=None):  # noqa: ARG001
        writer = DummyWriter()
        created.append(writer)
        return writer

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    def fake_to_excel(self, writer, sheet_name, index=False, **_: object) -> None:  # noqa: ARG002
        ws = DummyWorksheet("Temp")
        writer.book.worksheets.append(ws)
        writer.sheets[sheet_name] = ws
        writer.sheets["Temp"] = ws

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    df = pd.DataFrame({"value": [1]})
    out = tmp_path / "strip.xlsx"
    export.export_to_excel({"summary": df}, out)

    assert created, "Writer should be instantiated"
    writer = created[0]
    assert "Temp" not in writer.sheets
    assert "summary" in writer.sheets
    assert writer.book.worksheets[-1].title == "summary"


def test_flat_frames_from_results_collects_changes(monkeypatch):
    frames = pd.Series([1.0], name="value").to_frame()
    dummy_frames = {
        "summary": frames.copy(),
        "period_1": frames.copy(),
    }

    monkeypatch.setattr(
        export,
        "workbook_frames_from_results",
        lambda results: dummy_frames,
    )
    monkeypatch.setattr(
        export,
        "manager_contrib_table",
        lambda results: pd.DataFrame(
            {
                "Manager": ["M"],
                "Years": [1.0],
                "OOS CAGR": [0.1],
                "Contribution Share": [0.5],
            }
        ),
    )

    results = [
        {
            "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
            "manager_changes": [
                {
                    "action": "add",
                    "manager": "M",
                    "firm": "F",
                    "reason": None,
                    "detail": "",
                }
            ],
        }
    ]

    out = export.flat_frames_from_results(results)
    assert "periods" in out and "summary" in out
    assert "execution_metrics" in out
    assert "manager_contrib" in out
    assert "changes" in out and out["changes"].iloc[0]["Period"] == "2020-12"


def test_export_phase1_workbook_passes_summary_extensions(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_phase1_workbook_data(results, include_metrics=False):  # noqa: ARG001
        return {
            "summary": pd.DataFrame(),
            "period_1": pd.DataFrame(),
        }

    monkeypatch.setattr(export, "phase1_workbook_data", fake_phase1_workbook_data)
    monkeypatch.setattr(export, "reset_formatters_excel", lambda: None)

    def fake_make_period(sheet, res, in_s, in_e, out_s, out_e):  # noqa: ARG001
        captured.setdefault("period_calls", []).append((sheet, in_s, out_e))

    monkeypatch.setattr(export, "make_period_formatter", fake_make_period)

    def fake_manager_contrib(results):
        return pd.DataFrame(
            {
                "Manager": ["M"],
                "Years": [1.0],
                "OOS CAGR": [0.1],
                "Contribution Share": [0.5],
            }
        )

    monkeypatch.setattr(export, "manager_contrib_table", fake_manager_contrib)

    def fake_combined(results):
        return {"foo": "bar"}

    monkeypatch.setattr(export, "combined_summary_result", fake_combined)

    def record_summary(res, *args):  # noqa: ARG001
        captured["summary_res"] = res

    monkeypatch.setattr(export, "make_summary_formatter", record_summary)
    monkeypatch.setattr(
        export, "export_to_excel", lambda data, path: captured.setdefault("data", data)
    )

    results = [
        {
            "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
            "manager_changes": [
                {
                    "action": "add",
                    "manager": "M",
                    "firm": "F",
                    "reason": "r",
                    "detail": "d",
                }
            ],
        },
        {
            "period": ("2021-01", "2021-06", "2021-07", "2021-12"),
            "manager_changes": [
                {
                    "action": "drop",
                    "manager": "N",
                    "firm": "F2",
                    "reason": None,
                    "detail": None,
                }
            ],
        },
    ]

    export.export_phase1_workbook(results, tmp_path / "out.xlsx")

    assert captured.get("period_calls")
    summary_res = captured["summary_res"]
    assert len(summary_res["manager_changes"]) == 2
    assert "manager_contrib" in summary_res


def test_openpyxl_proxy_handles_formatting_helpers(monkeypatch):
    ws = DummyWorksheet("Report")
    proxy = export._OpenpyxlWorksheetProxy(ws)

    fmt = {"num_format": "0.00", "font_color": "red"}
    proxy.write(0, 0, "value", fmt)
    proxy.write_row(1, 0, [1, 2])

    first_cell = ws.cell(1, 1)
    assert first_cell.value == "value"
    assert first_cell.number_format == "0.00"
    assert first_cell.font.color == "FFFF0000"

    monkeypatch.setattr(
        export, "get_column_letter", lambda idx: chr(ord("A") + idx - 1)
    )
    proxy.set_column(0, 1, 18)
    assert ws.column_dimensions["A"].width == 18
    assert ws.column_dimensions["B"].width == 18

    proxy.freeze_panes(1, 1)
    assert ws.freeze_panes is ws.cell(2, 2)

    proxy.autofilter(0, 0, 1, 1)
    assert ws.auto_filter.ref == "A1:B2"

    monkeypatch.setattr(export, "get_column_letter", None, raising=False)
    proxy.set_column(0, 0, 10)
    proxy.autofilter(0, 0, 0, 0)
    assert ws.column_dimensions["A"].width == 18
    assert ws.auto_filter.ref == "A1:B2"


def test_openpyxl_workbook_adapter_prunes_and_proxies(monkeypatch):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class QuirkyWorkbook(DummyWorkbook):
        def create_sheet(self, title: str) -> DummyWorksheet:
            ws = super().create_sheet(title)
            ws.title = f"temp_{title}"
            return ws

    wb = QuirkyWorkbook()
    adapter = export._OpenpyxlWorkbookAdapter(wb)

    assert wb.worksheets == []

    ws_adapter = adapter.add_worksheet("Summary")
    assert isinstance(ws_adapter, export._OpenpyxlWorksheetAdapter)
    assert ws_adapter.native.title == "Summary"

    assert adapter.add_format(None) == {}
    fmt = adapter.add_format({"bold": True})
    assert fmt == {"bold": True}

    adapter.rename_last_sheet("Final")
    assert wb.worksheets[-1].title == "Final"
    assert adapter.foo == "bar"


def test_openpyxl_worksheet_adapter_exposes_native(monkeypatch):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    ws = DummyWorksheet("Data")
    adapter = export._OpenpyxlWorksheetAdapter(ws)
    assert adapter.native is ws


def test_export_to_excel_removes_default_and_renames(monkeypatch, tmp_path):
    utils_mod = SimpleNamespace(get_column_letter=lambda idx: chr(ord("A") + idx - 1))
    monkeypatch.setitem(sys.modules, "openpyxl", SimpleNamespace(utils=utils_mod))
    monkeypatch.setitem(sys.modules, "openpyxl.utils", utils_mod)

    class Writer:
        def __init__(self) -> None:
            self.book = DummyWorkbook()
            self.sheets: dict[str, object] = {"Sheet": object()}
            self.engine = "openpyxl"

        def __enter__(self) -> "Writer":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

    created: list[Writer] = []

    def fake_writer(path, engine=None, **_):  # noqa: ARG001
        if engine == "xlsxwriter":
            raise ModuleNotFoundError
        writer = Writer()
        created.append(writer)
        return writer

    monkeypatch.setattr(pd, "ExcelWriter", fake_writer)

    def fake_to_excel(self, writer, sheet_name, index=False, **_):  # noqa: ARG002
        ws = DummyWorksheet("TempName")
        writer.book.worksheets.append(ws)

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=False)

    df = pd.DataFrame({"value": [1.0]})
    export.export_to_excel({"summary": df}, tmp_path / "out.xlsx")

    assert created
    writer = created[0]
    assert all(ws.title != "Sheet" for ws in writer.book.worksheets)
    assert writer.book.worksheets[-1].title == "summary"
