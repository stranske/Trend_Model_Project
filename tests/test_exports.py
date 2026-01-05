import numpy as np
import pandas as pd
import pytest

import trend_analysis.export as export_module
from trend_analysis.export import (
    FORMATTERS_EXCEL,
    execution_metrics_frame,
    export_data,
    export_execution_metrics,
    export_to_excel,
    manager_contrib_table,
    register_formatter_excel,
    reset_formatters_excel,
)


def test_export_data(tmp_path):
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"B": [3, 4]})
    data = {"sheet1": df1, "sheet2": df2}
    out = tmp_path / "report"
    export_data(data, str(out), formats=["xlsx", "csv", "json", "txt"])

    assert (tmp_path / "report.xlsx").exists()
    assert (tmp_path / "report_sheet1.csv").exists()
    assert (tmp_path / "report_sheet2.csv").exists()
    assert (tmp_path / "report_sheet1.json").exists()
    assert (tmp_path / "report_sheet2.json").exists()
    assert (tmp_path / "report_sheet1.txt").exists()
    assert (tmp_path / "report_sheet2.txt").exists()

    read = pd.read_csv(tmp_path / "report_sheet1.csv")
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


def test_execution_metrics_frame_builds_labels_and_nan():
    results = [
        {
            "period": ("2024-01", "2024-06", "2024-07", "2024-12"),
            "turnover": 0.15,
            "transaction_cost": 0.002,
        },
        {
            "turnover": None,
        },
    ]

    frame = execution_metrics_frame(results)

    expected = pd.DataFrame(
        {
            "Period": ["2024-12", "period_2"],
            "Turnover": [0.15, np.nan],
            "Transaction Cost": [0.002, np.nan],
        }
    )

    pd.testing.assert_frame_equal(frame, expected)


def test_execution_metrics_frame_empty_results():
    frame = execution_metrics_frame([])
    assert frame.empty
    assert list(frame.columns) == ["Period", "Turnover", "Transaction Cost"]


def test_export_execution_metrics_delegates(monkeypatch):
    results_list = [
        {
            "period": ("2024-01", "2024-06", "2024-07", "2024-12"),
            "turnover": 0.1,
            "transaction_cost": 0.001,
        }
    ]
    captured: dict[str, object] = {}

    def fake_export_data(data, output_path, *, formats):
        captured["data"] = data
        captured["output_path"] = output_path
        captured["formats"] = tuple(formats)

    monkeypatch.setattr(export_module, "export_data", fake_export_data)

    export_execution_metrics((res for res in results_list), "out/report", formats=("csv", "json"))

    assert captured["output_path"] == "out/report"
    assert captured["formats"] == ("csv", "json")
    expected_df = execution_metrics_frame(results_list)
    actual_df = captured["data"]["execution_metrics"]
    assert isinstance(actual_df, pd.DataFrame)
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_manager_contrib_table_computes_participation():
    idx1 = pd.date_range("2024-01-31", periods=2, freq="ME")
    idx2 = pd.date_range("2024-03-31", periods=1, freq="ME")

    out1 = pd.DataFrame(
        {
            "FundA": [0.01, 0.02],
            "FundB": [0.005, 0.0],
            "FundSkip": [0.03, -0.01],
        },
        index=idx1,
    )
    out2 = pd.DataFrame(
        {
            "FundA": [0.015],
            "FundC": [0.04],
            "FundB": [0.02],
        },
        index=idx2,
    )

    results = [
        {
            "out_sample_scaled": out1,
            "fund_weights": {"FundA": 0.5, "FundB": 0.5, "FundSkip": 0.0},
        },
        {
            "out_sample_scaled": out2,
            "fund_weights": {"FundA": 0.6, "FundC": 0.4, "FundB": 0.0},
        },
    ]

    table = manager_contrib_table(results)

    assert list(table["Manager"]) == ["FundA", "FundC", "FundB"]

    expected_years = [3 / 12, 1 / 12, 2 / 12]
    assert table["Years"].tolist() == pytest.approx(expected_years)

    def cagr(values: list[float]) -> float:
        arr = np.array(values, dtype=float)
        gross = float(np.prod(1.0 + arr))
        periods = arr.size
        return gross ** (12.0 / periods) - 1.0

    expected_cagrs = [
        cagr([0.01, 0.02, 0.015]),
        cagr([0.04]),
        cagr([0.005, 0.0]),
    ]
    assert table["OOS CAGR"].tolist() == pytest.approx(expected_cagrs)

    # Compute contribution totals from test data to avoid magic numbers
    contrib_totals = {}
    for result in results:
        out_sample = result["out_sample_scaled"]
        weights = result["fund_weights"]
        for fund in out_sample.columns:
            contrib = (out_sample[fund] * weights.get(fund, 0.0)).sum()
            contrib_totals[fund] = contrib_totals.get(fund, 0.0) + contrib
    # Only include funds that appear in the output table
    contrib_totals = {k: contrib_totals[k] for k in ["FundA", "FundC", "FundB"]}
    total = sum(contrib_totals.values())
    expected_shares = [contrib_totals[name] / total for name in ["FundA", "FundC", "FundB"]]
    assert table["Contribution Share"].tolist() == pytest.approx(expected_shares)
    assert table["Contribution Share"].sum() == pytest.approx(1.0)


def test_manager_contrib_table_empty_results():
    table = manager_contrib_table([{"out_sample_scaled": pd.DataFrame(), "fund_weights": {}}])
    assert table.empty
    assert list(table.columns) == ["Manager", "Years", "OOS CAGR", "Contribution Share"]
