"""Additional coverage tests for export helpers."""

from __future__ import annotations

import math
import sys
import types
from collections import OrderedDict
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pytest

from trend_analysis import export as export_module

try:  # pragma: no cover - exercised when matplotlib is installed
    import matplotlib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled in test environment
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *args, **kwargs: None

    class _PyplotModule(types.ModuleType):
        """Minimal pyplot stub exposing the helpers under test."""

        def figure(self, *args: object, **kwargs: object) -> "_Figure":
            return _Figure()

        def close(self, *args: object, **kwargs: object) -> None:
            pass

    class _Axis:
        def plot(self, *args, **kwargs):
            return None

        def set_axis_off(self) -> None:
            return None

        def set_title(self, *args, **kwargs) -> None:
            return None

    class _Figure:
        def add_subplot(self, *args, **kwargs):
            return _Axis()

        def savefig(self, *args, **kwargs) -> None:
            return None

    pyplot = _PyplotModule("matplotlib.pyplot")

    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

from trend_analysis.export import (
    _format_frequency_policy_line,
    _OpenpyxlWorkbookAdapter,
    _OpenpyxlWorksheetProxy,
    export_multi_period_metrics,
    export_phase1_multi_metrics,
    export_phase1_workbook,
    flat_frames_from_results,
    format_summary_text,
    manager_contrib_table,
    phase1_workbook_data,
    summary_frame_from_result,
    workbook_frames_from_results,
)
from trend_analysis.pipeline import _compute_stats, calc_portfolio_returns


def _make_period_result(
    label: str,
    *,
    include_changes: bool = False,
    period_metadata: bool = True,
) -> dict[str, object]:
    """Return a synthetic result dictionary exercising export helpers."""

    idx_in = pd.date_range("2020-01-31", periods=3, freq="ME")
    idx_out = pd.date_range("2020-04-30", periods=2, freq="ME")
    in_df = pd.DataFrame(
        {
            "FundA": [0.01, 0.015, 0.02],
            "FundB": [0.005, 0.0, -0.002],
        },
        index=idx_in,
    )
    out_df = pd.DataFrame(
        {
            "FundA": [0.012, 0.014],
            "FundB": [0.001, -0.002],
        },
        index=idx_out,
    )

    funds = list(in_df.columns)
    ew_weights = {f: 1.0 / len(funds) for f in funds}
    fund_weights = {"FundA": 0.6, "FundB": 0.4}

    rf_in = pd.Series(0.0, index=idx_in)
    rf_out = pd.Series(0.0, index=idx_out)

    # Per-fund statistics reused by summary helpers
    fund_in_stats = _compute_stats(in_df, rf_in)
    fund_out_stats = _compute_stats(out_df, rf_out)

    def _portfolio_stats(
        weights: dict[str, float], frame: pd.DataFrame, rf: pd.Series
    ) -> dict[str, object]:
        arr = np.array([weights.get(col, 0.0) for col in frame.columns], dtype=float)
        returns = calc_portfolio_returns(arr, frame)
        return _compute_stats(pd.DataFrame({"p": returns}), rf)["p"]

    in_ew_stats = _portfolio_stats(ew_weights, in_df, rf_in)
    out_ew_stats = _portfolio_stats(ew_weights, out_df, rf_out)
    in_user_stats = _portfolio_stats(fund_weights, in_df, rf_in)
    out_user_stats = _portfolio_stats(fund_weights, out_df, rf_out)

    period = (
        (
            str(idx_in[0].date()),
            str(idx_in[-1].date()),
            str(idx_out[0].date()),
            f"Period {label}",
        )
        if period_metadata
        else None
    )

    result: dict[str, object] = {
        "period": period,
        "in_sample_scaled": in_df,
        "out_sample_scaled": out_df,
        "ew_weights": ew_weights,
        "fund_weights": fund_weights,
        "in_ew_stats": in_ew_stats,
        "out_ew_stats": out_ew_stats,
        "in_user_stats": in_user_stats,
        "out_user_stats": out_user_stats,
        "in_sample_stats": fund_in_stats,
        "out_sample_stats": fund_out_stats,
        "benchmark_ir": {
            "Bench": {
                "FundA": 0.1 + 0.01 * int(label),
                "FundB": 0.05,
                "equal_weight": 0.02,
                "user_weight": 0.03,
            }
        },
    }

    if include_changes:
        result["manager_changes"] = [
            {"action": "add", "manager": "FundA", "detail": f"{label}"},
            {"action": "trim", "manager": "FundB"},
        ]

    return result


def test_flat_frames_from_results_includes_contrib_and_changes():
    result = _make_period_result("1", include_changes=True)
    frames = flat_frames_from_results([result])

    assert "manager_contrib" in frames
    assert not frames["manager_contrib"].empty
    assert "execution_metrics" in frames
    assert "changes" in frames
    # Manager change rows should include the generated period label
    assert set(frames["changes"]["Period"]) == {"Period 1"}


def test_phase1_workbook_data_adds_metrics_summary():
    results = [_make_period_result("1"), _make_period_result("2")]
    frames = phase1_workbook_data(results, include_metrics=True)

    assert "summary" in frames
    assert "metrics_Period 1" in frames
    assert "metrics_Period 2" in frames
    assert "metrics_summary" in frames


def test_workbook_frames_from_results_adds_summary_and_execution():
    results = [_make_period_result("1"), _make_period_result("2")]
    frames = workbook_frames_from_results(results)

    assert set(frames) >= {"summary", "execution_metrics", "Period 1", "Period 2"}
    assert not frames["execution_metrics"].empty


def test_export_phase1_workbook_missing_period_metadata(monkeypatch, tmp_path):
    results = [
        _make_period_result("1", include_changes=True, period_metadata=False),
        _make_period_result("2", period_metadata=False),
    ]

    period_calls: list[tuple[str, str, str, str, str]] = []
    summary_call: dict[str, object] = {}
    frames_written: dict[str, pd.DataFrame] = {}

    def fake_make_period_formatter(sheet, res, in_s, in_e, out_s, out_e):
        period_calls.append((sheet, in_s, in_e, out_s, out_e))
        return lambda ws, wb: None

    def fake_make_summary_formatter(res, in_s, in_e, out_s, out_e):
        summary_call["args"] = (in_s, in_e, out_s, out_e)
        summary_call["payload"] = res
        return lambda ws, wb: None

    def fake_export_to_excel(data, path):
        frames_written.update(data)

    monkeypatch.setattr(
        "trend_analysis.export.make_period_formatter", fake_make_period_formatter
    )
    monkeypatch.setattr(
        "trend_analysis.export.make_summary_formatter", fake_make_summary_formatter
    )
    monkeypatch.setattr("trend_analysis.export.export_to_excel", fake_export_to_excel)

    export_phase1_workbook(results, str(tmp_path / "out.xlsx"))

    # Missing period metadata should yield generated sheet names and blank ranges
    assert {call[0] for call in period_calls} == {"period_1", "period_2"}
    assert all(call[1:] == ("", "", "", "") for call in period_calls)

    assert summary_call["args"] == ("", "", "", "")
    payload = summary_call["payload"]
    assert "manager_contrib" in payload

    # Summary sheet should be first when exporting
    assert list(frames_written) and list(frames_written)[0] == "summary"


def test_export_phase1_workbook_collects_manager_changes(monkeypatch, tmp_path):
    results = [
        _make_period_result("1", include_changes=True),
        _make_period_result("2", include_changes=True),
    ]

    summary_call: dict[str, object] = {}

    def fake_make_summary_formatter(res, in_s, in_e, out_s, out_e):
        summary_call["args"] = (in_s, in_e, out_s, out_e)
        summary_call["payload"] = res
        return lambda ws, wb: None

    monkeypatch.setattr(
        "trend_analysis.export.make_summary_formatter", fake_make_summary_formatter
    )
    monkeypatch.setattr(
        "trend_analysis.export.export_to_excel", lambda data, path: None
    )

    export_phase1_workbook(results, str(tmp_path / "out.xlsx"))

    args = summary_call["args"]
    assert args == (
        results[0]["period"][0],
        results[0]["period"][1],
        results[-1]["period"][2],
        results[-1]["period"][3],
    )

    payload = summary_call["payload"]
    assert "manager_changes" in payload
    assert len(payload["manager_changes"]) == 4
    assert "manager_contrib" in payload


@pytest.mark.parametrize(
    "exporter",
    [export_phase1_multi_metrics, export_multi_period_metrics],
)
def test_export_multi_helpers_include_metrics(tmp_path, exporter):
    results = [_make_period_result("1"), _make_period_result("2")]
    out = tmp_path / "report"

    exporter(results, str(out), formats=["csv"], include_metrics=True)

    metrics_path = out.with_name(f"{out.stem}_metrics.csv")
    summary_path = out.with_name(f"{out.stem}_metrics_summary.csv")

    assert metrics_path.exists()
    assert summary_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    assert set(metrics_df["Period"]) == {"Period 1", "Period 2"}

    summary_df = pd.read_csv(summary_path)
    assert not summary_df.empty


def test_export_phase1_multi_metrics_all_formats(monkeypatch, tmp_path):
    """Ensure excel + flat exports both run and include metric tables."""

    results = [_make_period_result("1", include_changes=True), _make_period_result("2")]

    workbook_calls: list[tuple[str, bool, int]] = []
    data_calls: list[tuple[str, tuple[str, ...], dict[str, pd.DataFrame]]] = []

    def fake_workbook(res_list, path, include_metrics=False):
        workbook_calls.append((path, include_metrics, len(res_list)))

    def fake_export_data(data, output_path, *, formats):
        data_calls.append((output_path, tuple(formats), data))

    monkeypatch.setattr(export_module, "export_phase1_workbook", fake_workbook)
    monkeypatch.setattr(export_module, "export_data", fake_export_data)

    export_phase1_multi_metrics(
        results, str(tmp_path / "report"), formats=["xlsx", "csv"], include_metrics=True
    )

    assert workbook_calls == [
        (str((tmp_path / "report").with_suffix(".xlsx")), True, len(results))
    ]
    assert data_calls, "Expected non-excel export to be invoked"
    _, formats, payload = data_calls[0]
    assert formats == ("csv",)
    # Expect consolidated metrics plus summary + manager tables when available
    assert {
        "periods",
        "summary",
        "metrics",
        "metrics_summary",
        "manager_contrib",
    }.issubset(payload)
    metrics = payload["metrics"]
    assert "Period" in metrics.columns


def test_export_multi_period_metrics_all_formats(monkeypatch, tmp_path):
    """Exercise both excel + flat exporters including summary helpers."""

    results = [_make_period_result("1", include_changes=True), _make_period_result("2")]

    period_calls: list[tuple[str, str, str, str, str]] = []
    summary_calls: list[tuple[dict[str, object], tuple[str, str, str, str]]] = []
    export_calls: list[tuple[tuple[str, ...], str, dict[str, pd.DataFrame]]] = []

    def fake_make_period(sheet, res, in_s, in_e, out_s, out_e):
        period_calls.append((sheet, in_s, in_e, out_s, out_e))
        return lambda ws, wb: None

    def fake_make_summary(res, in_s, in_e, out_s, out_e):
        summary_calls.append((res, (in_s, in_e, out_s, out_e)))
        return lambda ws, wb: None

    def fake_export_data(data, output_path, *, formats):
        export_calls.append((tuple(formats), output_path, data))

    monkeypatch.setattr(export_module, "make_period_formatter", fake_make_period)
    monkeypatch.setattr(export_module, "make_summary_formatter", fake_make_summary)
    monkeypatch.setattr(export_module, "export_data", fake_export_data)

    export_multi_period_metrics(
        results,
        str(tmp_path / "multi"),
        formats=["xlsx", "json"],
        include_metrics=True,
    )

    # Period formatters should be created for both sheets with populated ranges
    assert {call[0] for call in period_calls} == {"Period 1", "Period 2"}
    assert all(call[1:] != ("", "", "", "") for call in period_calls)

    # Summary formatter should be invoked with manager contribution payload
    assert summary_calls
    payload, ranges = summary_calls[0]
    assert ranges[0] and ranges[3], "Expected populated in/out ranges"
    assert "manager_contrib" in payload

    # Both excel and json exporters should be triggered with enriched data
    assert {formats for formats, _, _ in export_calls} == {("xlsx",), ("json",)}
    excel_payload = next(data for fmt, _, data in export_calls if fmt == ("xlsx",))
    assert "summary" in excel_payload and "metrics_summary" in excel_payload
    json_payload = next(data for fmt, _, data in export_calls if fmt == ("json",))
    assert {"periods", "summary", "metrics", "metrics_summary"}.issubset(json_payload)


def test_summary_frame_from_result_handles_mixed_stats():
    tuple_stats = (0.05, 0.10, 1.2, 0.9, 0.7, 0.15)
    obj_stats = types.SimpleNamespace(
        cagr=0.06,
        vol=0.12,
        sharpe=0.95,
        sortino=0.85,
        information_ratio=0.55,
        max_drawdown=0.20,
    )
    result = {
        "in_ew_stats": tuple_stats,
        "out_ew_stats": obj_stats,
        "in_user_stats": tuple_stats,
        "out_user_stats": obj_stats,
        "in_sample_stats": {"FundA": tuple_stats},
        "out_sample_stats": {"FundA": obj_stats},
        "fund_weights": {"FundA": 0.5},
        "benchmark_ir": {
            "Bench": {"FundA": 0.4, "equal_weight": 0.2, "user_weight": 0.25}
        },
    }

    frame = export_module.summary_frame_from_result(result)

    assert {"OS IR Bench", "OS MaxDD"}.issubset(frame.columns)
    fund_row = frame[frame["Name"] == "FundA"].iloc[0]
    assert fund_row["Weight"] == pytest.approx(50.0)
    assert fund_row["OS IR Bench"] == pytest.approx(20.0)
    assert fund_row["OS MaxDD"] == pytest.approx(0.4)


def test_manager_contrib_table_computes_shares_and_handles_empty():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    out_df = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.0],
            "FundB": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    results = [
        {"out_sample_scaled": out_df, "fund_weights": {"FundA": 0.6, "FundB": 0.4}}
    ]

    table = export_module.manager_contrib_table(results)

    assert set(table["Manager"]) == {"FundA", "FundB"}
    assert table.loc[table["Manager"] == "FundB", "Contribution Share"].iloc[
        0
    ] == pytest.approx(0.0)
    assert table.loc[table["Manager"] == "FundA", "Contribution Share"].iloc[0] > 0

    empty = export_module.manager_contrib_table(
        [{"out_sample_scaled": pd.DataFrame(), "fund_weights": {}}]
    )
    assert empty.empty


def test_format_summary_text_formats_ints():
    stat = types.SimpleNamespace(
        cagr=0.1,
        vol=0.2,
        sharpe=1.4,
        sortino=1.0,
        information_ratio=1,
        max_drawdown=-0.25,
    )
    res = {
        "in_ew_stats": stat,
        "out_ew_stats": stat,
        "in_user_stats": stat,
        "out_user_stats": stat,
        "in_sample_stats": {"FundA": stat},
        "out_sample_stats": {"FundA": stat},
        "fund_weights": {"FundA": 1},
        "benchmark_ir": {
            "Bench": {"FundA": 0.5, "equal_weight": 0.2, "user_weight": 0.3}
        },
    }

    text = format_summary_text(res, "2020-01", "2020-06", "2020-07", "2020-12")
    assert "Equal Weight" in text
    assert "1.00" in text


def test_manager_contrib_table_handles_sparse_series(monkeypatch):
    results = [
        {
            "out_sample_scaled": pd.DataFrame(
                {"FundA": [0.01, 0.0], "FundB": [np.nan, np.nan]}
            ),
            "fund_weights": {"FundA": 0.6, "FundB": 0.4},
        },
        {
            "out_sample_scaled": pd.DataFrame({"FundA": [0.02, 0.03]}),
            "fund_weights": {"FundA": 0.5},
        },
    ]

    real_concat = pd.concat

    def fake_concat(objs, *args, **kwargs):  # noqa: ANN001
        fake_concat.calls += 1
        if fake_concat.calls == 2:
            return pd.Series(dtype=float)
        return real_concat(objs, *args, **kwargs)

    fake_concat.calls = 0
    monkeypatch.setattr(pd, "concat", fake_concat)

    table = manager_contrib_table(results)
    assert not table.empty
    assert table["Contribution Share"].sum() <= 1.0


def test_manager_contrib_table_sets_nan_when_concat_empty(monkeypatch):
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    out_df = pd.DataFrame({"FundA": [0.01, 0.02]}, index=idx)
    results = [{"out_sample_scaled": out_df, "fund_weights": {"FundA": 1.0}}]

    real_concat = pd.concat

    def fake_concat(objs, *args, **kwargs):  # noqa: ANN001
        fake_concat.calls += 1
        if fake_concat.calls == 1:
            return pd.Series(dtype=float)
        return real_concat(objs, *args, **kwargs)

    fake_concat.calls = 0
    monkeypatch.setattr(pd, "concat", fake_concat)

    table = manager_contrib_table(results)
    assert math.isnan(float(table.loc[0, "OOS CAGR"]))


def test_workbook_frames_from_results_includes_summary():
    frames = workbook_frames_from_results([_make_period_result("1")])
    assert "summary" in frames


def test_workbook_frames_from_results_adds_execution_metrics(monkeypatch):
    summary_df = pd.DataFrame({"x": [1]})
    metrics_df = pd.DataFrame({"m": [2]})

    monkeypatch.setattr(
        export_module,
        "summary_frame_from_result",
        lambda res: summary_df.copy(),
    )
    monkeypatch.setattr(
        export_module,
        "combined_summary_result",
        lambda res: {"data": 1},
    )
    monkeypatch.setattr(
        export_module,
        "execution_metrics_frame",
        lambda res: metrics_df.copy(),
    )

    frames = workbook_frames_from_results(
        [{"period": ("2020-01", "2020-03", "2020-04", "Period 1")}]
    )

    assert "summary" in frames and "execution_metrics" in frames
    pd.testing.assert_frame_equal(frames["execution_metrics"], metrics_df)


def test_phase1_workbook_data_appends_metrics_summary(monkeypatch):
    base_frames = OrderedDict(
        [
            ("summary", pd.DataFrame({"x": [1]})),
            ("execution_metrics", pd.DataFrame({"m": [2]})),
        ]
    )

    monkeypatch.setattr(
        export_module,
        "workbook_frames_from_results",
        lambda results: base_frames.copy(),
    )
    monkeypatch.setattr(
        export_module,
        "metrics_from_result",
        lambda res: pd.DataFrame({"metric": [42]}),
    )
    monkeypatch.setattr(
        export_module,
        "combined_summary_result",
        lambda res: {"metrics": True},
    )

    frames = phase1_workbook_data(
        [{"period": ("2020-01", "2020-06", "2020-07", "Period 1")}],
        include_metrics=True,
    )

    assert "metrics_Period 1" in frames
    assert "metrics_summary" in frames


def test_flat_frames_from_results_includes_manager_tables(monkeypatch):
    base_frames = OrderedDict(
        [
            ("Period 1", pd.DataFrame({"val": [1]})),
            ("summary", pd.DataFrame({"s": [2]})),
            ("execution_metrics", pd.DataFrame({"m": [3]})),
        ]
    )

    monkeypatch.setattr(
        export_module,
        "workbook_frames_from_results",
        lambda results: base_frames.copy(),
    )
    monkeypatch.setattr(
        export_module,
        "manager_contrib_table",
        lambda results: pd.DataFrame(
            {
                "Manager": ["FundA"],
                "Years": [1.0],
                "OOS CAGR": [0.1],
                "Contribution Share": [0.5],
            }
        ),
    )
    monkeypatch.setattr(
        export_module,
        "execution_metrics_frame",
        lambda results: pd.DataFrame({"m": [3]}),
    )

    frames = flat_frames_from_results(
        [
            {
                "period": ("2020-01", "2020-06", "2020-07", "Period 1"),
                "manager_changes": [{"action": "add", "manager": "FundA"}],
            }
        ]
    )

    assert {
        "periods",
        "summary",
        "manager_contrib",
        "changes",
        "execution_metrics",
    }.issubset(frames)


def test_openpyxl_proxy_applies_formatting():
    class DummyFont:
        def __init__(self) -> None:
            self.color: str | None = None

        def copy(self, *, color: str) -> "DummyFont":
            new = DummyFont()
            new.color = color
            return new

    class DummyCell:
        def __init__(self) -> None:
            self.number_format: str | None = None
            self.font = DummyFont()

    proxy = _OpenpyxlWorksheetProxy(object())
    cell = DummyCell()

    proxy._apply_format(cell, {"num_format": "0.0", "font_color": "#ff0000"})

    assert cell.number_format == "0.0"
    assert cell.font.color == "FFFF0000"


def test_openpyxl_workbook_adapter_prunes_default_sheet():
    class DummySheet:
        def __init__(self) -> None:
            self.title = "Sheet"

        def cell(self, *_: object) -> "DummySheet":
            self.value = ""
            return self

    class DummyWorkbook:
        def __init__(self) -> None:
            self.worksheets = [DummySheet()]
            self.removed = False

        def remove(self, sheet: DummySheet) -> None:
            self.removed = True
            self.worksheets.remove(sheet)

        def create_sheet(self, title: str) -> DummySheet:
            sheet = DummySheet()
            sheet.title = title
            self.worksheets.append(sheet)
            return sheet

    wb = DummyWorkbook()
    adapter = _OpenpyxlWorkbookAdapter(wb)
    assert wb.removed is True
    ws = adapter.add_worksheet("Report")
    assert ws.native.title == "Report"


def test_maybe_remove_openpyxl_default_sheet_handles_multiple(monkeypatch):
    class DummySheet:
        def __init__(self, title: str, value: object | None) -> None:
            self.title = title
            self._value = value

        def cell(self, row: int, column: int) -> "DummySheet":  # noqa: ARG002
            return self

        @property
        def value(self) -> object | None:  # pragma: no cover - attribute access only
            return self._value

    class DummyWorkbook:
        def __init__(self) -> None:
            self.worksheets = [DummySheet("Summary", None), DummySheet("Sheet", None)]

        def remove(self, _: object) -> None:  # pragma: no cover - should not run
            raise AssertionError(
                "remove should not be called when multiple sheets exist"
            )

    workbook = DummyWorkbook()
    remover = getattr(export_module, "_maybe_remove_openpyxl_default_sheet")

    removed = remover(workbook)
    assert removed is None


def test_openpyxl_proxy_column_and_autofilter(monkeypatch):
    class DummyDimensions(dict):
        def __getitem__(self, key: object) -> "DummyDimension":
            dim = super().get(key)
            if dim is None:
                dim = DummyDimension()
                super().__setitem__(key, dim)
            return dim

    class DummyDimension:
        def __init__(self) -> None:
            self.width = 0.0

    class DummyAutoFilter:
        def __init__(self) -> None:
            self.ref = ""

    class DummyWorksheet:
        def __init__(self) -> None:
            self.column_dimensions: DummyDimensions = DummyDimensions()
            self.auto_filter = DummyAutoFilter()

        def cell(self, row: int, column: int) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                value=None,
                font=types.SimpleNamespace(copy=lambda **_: types.SimpleNamespace()),
            )

    ws = DummyWorksheet()
    proxy = _OpenpyxlWorksheetProxy(ws)

    monkeypatch.setattr(
        export_module, "get_column_letter", lambda idx: {2: "B", 3: "C"}[idx]
    )

    proxy.set_column(1, 2, 12.5)
    assert ws.column_dimensions["B"].width == pytest.approx(12.5)
    assert ws.column_dimensions["C"].width == pytest.approx(12.5)

    proxy.autofilter(0, 1, 4, 2)
    assert ws.auto_filter.ref == "B1:C5"


def test_openpyxl_workbook_adapter_rename_last_sheet():
    class DummySheet:
        def __init__(self, title: str, value: object | None = None) -> None:
            self.title = title
            self._value = value

        def cell(self, *_: object) -> SimpleNamespace:
            return SimpleNamespace(value=self._value)

    class DummyWorkbook:
        def __init__(self) -> None:
            self.worksheets = [DummySheet("Sheet")]

        def remove(self, sheet: DummySheet) -> None:
            self.worksheets.remove(sheet)

        def create_sheet(self, title: str) -> DummySheet:
            new_sheet = DummySheet(title)
            self.worksheets.append(new_sheet)
            return new_sheet

    wb = DummyWorkbook()
    adapter = _OpenpyxlWorkbookAdapter(wb)
    ws = adapter.add_worksheet("Metrics")
    assert ws.native.title == "Metrics"

    wb.worksheets.append(DummySheet("Old Name"))
    adapter.rename_last_sheet("Renamed")
    assert wb.worksheets[-1].title == "Renamed"


def test_format_frequency_policy_line_includes_extras():
    res = {
        "input_frequency": {
            "label": "Monthly",
            "target_label": "Quarterly",
            "resampled": True,
        },
        "missing_data_policy": {
            "policy": "ffill",
            "limit": 2,
            "total_filled": "3",
            "dropped_assets": ["A", "B"],
        },
    }

    line = _format_frequency_policy_line(res)
    assert "Monthly" in line and "Quarterly" in line
    assert "limit=2" in line
    assert "filled 3 cells" in line
    assert "dropped 2 assets" in line


def test_summary_frame_from_result_includes_diagnostics():
    stats_template = dict(
        cagr=0.1,
        vol=0.2,
        sharpe=1.1,
        sortino=0.9,
        information_ratio=0.4,
        max_drawdown=-0.3,
    )

    def _stat(**overrides: float) -> SimpleNamespace:
        return SimpleNamespace(**{**stats_template, **overrides})

    res: dict[str, object] = {
        "in_ew_stats": _stat(),
        "out_ew_stats": _stat(),
        "in_user_stats": _stat(),
        "out_user_stats": _stat(),
        "in_sample_stats": {"FundA": _stat(), "FundB": _stat()},
        "out_sample_stats": {"FundA": _stat(), "FundB": _stat()},
        "fund_weights": {"FundA": 0.6, "FundB": 0.4},
        "benchmark_ir": {
            "Bench": {"FundA": 0.2, "equal_weight": 0.1, "user_weight": 0.15}
        },
        "preprocessing_summary": "Preprocessing ok",
        "risk_diagnostics": {
            "asset_volatility": pd.DataFrame({"FundA": [0.12], "FundB": [0.05]}),
            "portfolio_volatility": pd.Series([0.08]),
            "turnover_value": 0.03,
        },
        "performance_by_regime": pd.DataFrame(
            {
                ("Portfolio", "Bull"): {
                    "CAGR": 0.1,
                    "Sharpe": 1.2,
                    "Max Drawdown": -0.2,
                    "Hit Rate": 0.6,
                    "Observations": 5,
                },
                ("Portfolio", "Bear"): {
                    "CAGR": math.nan,
                    "Sharpe": math.nan,
                    "Max Drawdown": math.nan,
                    "Hit Rate": math.nan,
                    "Observations": math.nan,
                },
            }
        ),
        "regime_summary": "Strong momentum",
        "regime_notes": [" Note A ", ""],
        "input_frequency": {"label": "Monthly"},
        "missing_data_policy": {},
    }

    frame = summary_frame_from_result(res)
    assert "OS IR Bench" in frame.columns
    assert "Equal Weight" in frame["Name"].tolist()
    fund_row = frame.loc[frame["Name"] == "FundA"].iloc[0]
    assert fund_row["Weight"] == pytest.approx(60.0)
    assert fund_row["OS IR Bench"] == pytest.approx(-30.0)
    assert fund_row["OS MaxDD"] == pytest.approx(0.2)


def test_export_phase1_workbook_without_summary(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_export(frames: Mapping[str, pd.DataFrame], path: str) -> None:
        captured["frames"] = dict(frames)
        captured["path"] = path

    monkeypatch.setattr(export_module, "export_to_excel", fake_export)

    export_phase1_workbook([], str(tmp_path / "empty.xlsx"))

    # With no results the workbook should still be exported with no summary sheet
    assert captured["frames"] == {}


def test_export_phase1_multi_metrics_skips_empty_metrics(monkeypatch, tmp_path):
    exported: list[tuple[tuple[str, ...], str, Mapping[str, pd.DataFrame]]] = []

    def fake_export(
        data: Mapping[str, pd.DataFrame], path: str, *, formats: Iterable[str]
    ):
        exported.append((tuple(formats), path, dict(data)))

    monkeypatch.setattr(export_module, "export_data", fake_export)

    export_phase1_multi_metrics(
        [],
        str(tmp_path / "phase1"),
        formats=["csv", "json"],
        include_metrics=True,
    )

    assert exported
    for formats, _, payload in exported:
        assert payload.get("metrics") is None or payload["metrics"].empty


def test_export_multi_period_metrics_handles_other_only(monkeypatch, tmp_path):
    calls: list[tuple[tuple[str, ...], Mapping[str, pd.DataFrame]]] = []

    def fake_export(
        data: Mapping[str, pd.DataFrame], _: str, *, formats: Iterable[str]
    ):
        calls.append((tuple(formats), dict(data)))

    monkeypatch.setattr(export_module, "export_data", fake_export)

    export_multi_period_metrics(
        [],
        str(tmp_path / "multi"),
        formats=["csv"],
        include_metrics=True,
    )

    assert len(calls) == 1
    formats, payload = calls[0]
    assert formats == ("csv",)
    assert "metrics" not in payload
    assert "metrics_summary" not in payload
