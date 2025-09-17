"""Additional coverage tests for export helpers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from trend_analysis import export as export_module

try:  # pragma: no cover - exercised when matplotlib is installed
    import matplotlib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled in test environment
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *args, **kwargs: None
    pyplot = types.ModuleType("matplotlib.pyplot")

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

    def _figure(*args, **kwargs):
        return _Figure()

    pyplot.figure = _figure
    pyplot.close = lambda *args, **kwargs: None

    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

from trend_analysis.export import (
    export_multi_period_metrics,
    export_phase1_multi_metrics,
    export_phase1_workbook,
    flat_frames_from_results,
    phase1_workbook_data,
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

    def _portfolio_stats(weights: dict[str, float], frame: pd.DataFrame, rf: pd.Series) -> dict[str, object]:
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

    monkeypatch.setattr("trend_analysis.export.make_period_formatter", fake_make_period_formatter)
    monkeypatch.setattr("trend_analysis.export.make_summary_formatter", fake_make_summary_formatter)
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

    monkeypatch.setattr("trend_analysis.export.make_summary_formatter", fake_make_summary_formatter)
    monkeypatch.setattr("trend_analysis.export.export_to_excel", lambda data, path: None)

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
    assert {"periods", "summary", "metrics", "metrics_summary", "manager_contrib"}.issubset(
        payload
    )
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
