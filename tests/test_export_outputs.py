import json
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.export import (
    FORMATTERS_EXCEL,
    export_to_csv,
    export_to_excel,
    export_to_json,
    export_to_txt,
)

pytestmark = pytest.mark.cosmetic


@pytest.fixture(autouse=True)
def restore_formatters():
    """Ensure tests do not leak formatter registrations."""
    original = FORMATTERS_EXCEL.copy()
    FORMATTERS_EXCEL.clear()
    try:
        yield
    finally:
        FORMATTERS_EXCEL.clear()
        FORMATTERS_EXCEL.update(original)


class DummyWorksheet:
    def __init__(self, name: str):
        self.name = name
        self.writes: list[tuple[int, int, object]] = []

    def write(self, row: int, col: int, value: object, fmt=None) -> None:
        self.writes.append((row, col, value))


class DummyWorkbook:
    # This method is intended to mock the 'add_worksheet' method from the real workbook class
    # (e.g., xlsxwriter.Workbook), but the signature differs for testing purposes.
    def add_worksheet(self, name: str) -> DummyWorksheet:  # type: ignore[override]
        return DummyWorksheet(name)

    def add_format(self, _options=None):
        return object()


class DummyWriter:
    def __init__(self, path: Path, *, engine: str | None):
        self.path = Path(path)
        self.engine = engine
        self.book = DummyWorkbook()
        self.sheets: dict[str, DummyWorksheet] = {}

    def __enter__(self) -> "DummyWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_export_to_excel_falls_back_without_xlsxwriter(monkeypatch, tmp_path):
    calls: list[tuple[Path, str | None]] = []

    def fake_excel_writer(path, engine=None):
        calls.append((Path(path), engine))
        if engine == "xlsxwriter":
            raise ModuleNotFoundError("xlsxwriter unavailable")
        return DummyWriter(Path(path), engine=engine)

    monkeypatch.setattr(pd, "ExcelWriter", fake_excel_writer)

    out_path = tmp_path / "nested" / "report.xlsx"
    sheet_calls: list[str] = []

    def sheet_formatter(ws, wb):
        sheet_calls.append(ws.name)
        ws.write(0, 0, "ok")

    export_to_excel(
        {"summary": pd.DataFrame({"A": [1]})},
        str(out_path),
        default_sheet_formatter=sheet_formatter,
    )

    assert calls == [(out_path, "xlsxwriter"), (out_path, None)]
    assert sheet_calls == ["summary"]
    assert out_path.parent.exists()


def test_export_helpers_apply_formatter_and_create_outputs(tmp_path):
    df = pd.DataFrame({"A": [1], "B": [2]})

    def formatter(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["C"] = out["A"] + out["B"]
        return out

    csv_base = tmp_path / "data" / "export.csv"
    export_to_csv({"sheet": df}, str(csv_base), formatter=formatter)
    csv_path = csv_base.with_name("export_sheet.csv")
    assert csv_path.exists()
    csv_content = csv_path.read_text()
    assert "C" in csv_content and "3" in csv_content

    json_base = tmp_path / "data" / "export.json"
    export_to_json({"sheet": df}, str(json_base), formatter=formatter)
    json_path = json_base.with_name("export_sheet.json")
    data = json.loads(json_path.read_text())
    assert data[0]["C"] == 3

    txt_base = tmp_path / "data" / "export.txt"
    export_to_txt({"sheet": df}, str(txt_base), formatter=formatter)
    txt_path = txt_base.with_name("export_sheet.txt")
    txt_content = txt_path.read_text()
    assert "C" in txt_content and "3" in txt_content
