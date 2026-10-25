"""Tests for ``streamlit_app.components.comparison_export`` (WP t9)."""

from __future__ import annotations

import json
import math
import zipfile
from io import BytesIO, StringIO
from types import SimpleNamespace
from typing import Any

import openpyxl
import pandas as pd
import pytest

from streamlit_app.components import comparison_export as ce


def _result(
    *,
    user_stats: Any = None,
    metrics: Any = None,
    out_sample: Any = None,
    period_results: Any = None,
    include_user: bool = True,
) -> SimpleNamespace:
    details: dict[str, Any] = {}
    if include_user:
        details["out_user_stats"] = user_stats
    if out_sample is not None:
        details["out_sample_stats"] = out_sample
    if period_results is not None:
        details["period_results"] = period_results
    return SimpleNamespace(details=details, metrics=metrics)


def _period(
    start: str = "2024-01",
    end: str = "2024-03",
    funds: list[str] | None = None,
    weights: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "period": ("", "", start, end),
        "selected_funds": funds or [],
        "fund_weights": weights or {},
    }
    d.update(extra)
    return d


# ---------------------------------------------------------------- constants


def test_constants_exact() -> None:
    assert ce.HEADER_FORMAT == {
        "bold": True,
        "bg_color": "#4472C4",
        "font_color": "white",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    }
    assert ce.PERCENT_FORMAT == "0.00%"
    assert ce.RATIO_FORMAT == "0.00"
    assert ce.NUMBER_FORMAT == "#,##0.00"
    assert ce.DATE_FORMAT == "yyyy-mm-dd"
    assert ce.WINNER_GREEN == "#C6EFCE"
    assert ce.LOSER_RED == "#FFC7CE"
    assert ce.NEUTRAL_YELLOW == "#FFEB9C"


# ---------------------------------------------------------------- _coerce_numeric


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, 2.0),
        ("2.5", 2.5),
        (-0.25, -0.25),
        (float("inf"), None),
        (float("-inf"), None),
        (float("nan"), None),
        (None, None),
        ("abc", None),
        ([1], None),
    ],
)
def test_coerce_numeric(value: Any, expected: float | None) -> None:
    result = ce._coerce_numeric(value)
    if expected is None:
        assert result is None
    else:
        assert result == expected
        assert isinstance(result, float)


# ---------------------------------------------------------------- _extract_metrics


def test_extract_metrics_from_object_user_stats() -> None:
    stats = SimpleNamespace(sharpe=1.5, cagr="0.12", _hidden=99, label="x")
    res = _result(user_stats=stats, metrics=pd.DataFrame([{"Sharpe": 9.0}]))
    out = ce._extract_metrics(res)
    assert out == {"sharpe": 1.5, "cagr": 0.12, "label": None}
    assert "_hidden" not in out


def test_extract_metrics_from_dict_user_stats_priority() -> None:
    res = _result(
        user_stats={"Sharpe": "2.0", "Bad": "zz"},
        metrics=pd.DataFrame([{"Sharpe": 9.0, "Other": 1.0}]),
    )
    assert ce._extract_metrics(res) == {"Sharpe": 2.0, "Bad": None}


def test_extract_metrics_empty_user_stats_falls_to_metrics_df() -> None:
    res = _result(
        user_stats={},
        metrics=pd.DataFrame([{"Sharpe": 1.0, "CAGR": 0.1}, {"Sharpe": 5.0, "CAGR": 5.0}]),
    )
    # Only first row of metrics DataFrame is used.
    assert ce._extract_metrics(res) == {"Sharpe": 1.0, "CAGR": 0.1}


def test_extract_metrics_empty_metrics_falls_to_out_sample_dict() -> None:
    res = _result(
        user_stats=None,
        metrics=pd.DataFrame(),
        out_sample={"Vol": 0.3, "Ret": "0.05"},
    )
    assert ce._extract_metrics(res) == {"Vol": 0.3, "Ret": 0.05}


def test_extract_metrics_out_sample_object() -> None:
    res = _result(
        user_stats=None,
        metrics=None,
        out_sample=SimpleNamespace(alpha=0.7, beta="bad"),
    )
    assert ce._extract_metrics(res) == {"alpha": 0.7, "beta": None}


def test_extract_metrics_no_data_returns_empty() -> None:
    assert ce._extract_metrics(_result(user_stats=None, metrics=None)) == {}
    assert ce._extract_metrics(SimpleNamespace(details=None, metrics=None)) == {}
    assert ce._extract_metrics(SimpleNamespace()) == {}


def test_extract_metrics_non_dataframe_metrics_ignored() -> None:
    res = _result(user_stats=None, metrics={"Sharpe": 1.0}, out_sample={"X": 3})
    assert ce._extract_metrics(res) == {"X": 3.0}


# ---------------------------------------------------------------- _extract_period_results / _determine_winner


def test_extract_period_results_exact() -> None:
    periods = [{"period": ("", "", "2024-01", "2024-03")}]
    assert ce._extract_period_results(_result(period_results=periods)) == periods
    assert ce._extract_period_results(_result(period_results=None)) == []
    assert ce._extract_period_results(SimpleNamespace(details={"period_results": None})) == []
    assert ce._extract_period_results(SimpleNamespace(details=None)) == []
    assert ce._extract_period_results(SimpleNamespace()) == []


@pytest.mark.parametrize(
    ("a", "b", "higher", "expected"),
    [
        (None, 1.0, True, ""),
        (1.0, None, True, ""),
        (None, None, True, ""),
        (1.0, 1.0, True, "Tie"),
        (1.0, 1.0 + 5e-11, True, "Tie"),
        (2.0, 1.0, True, "A"),
        (1.0, 2.0, True, "B"),
        (1.0, 2.0, False, "A"),
        (2.0, 1.0, False, "B"),
        (1.0, 1.0, False, "Tie"),
    ],
)
def test_determine_winner(a: Any, b: Any, higher: bool, expected: str) -> None:
    assert ce._determine_winner(a, b, higher_is_better=higher) == expected


def test_determine_winner_default_higher_is_better() -> None:
    assert ce._determine_winner(3.0, 1.0) == "A"
    assert ce._determine_winner(1.0, 3.0) == "B"


# ---------------------------------------------------------------- build_comparison_summary_df


def test_summary_df_exact_values_and_winners() -> None:
    res_a = _result(user_stats={"Sharpe": 2.0, "Volatility": 0.10, "Alpha": 1.0, "Zero": 0.0})
    res_b = _result(user_stats={"Sharpe": 1.0, "Volatility": 0.20, "Beta": 5.0, "Zero": 0.5})
    df = ce.build_comparison_summary_df(res_a, res_b, label_a="CfgA", label_b="CfgB")
    assert list(df.columns) == ["Metric", "CfgA", "CfgB", "Delta (B-A)", "% Change", "Winner"]
    assert list(df["Metric"]) == ["Alpha", "Beta", "Sharpe", "Volatility", "Zero"]
    by = df.set_index("Metric")
    # Alpha only in A
    assert by.loc["Alpha", "CfgA"] == 1.0
    assert math.isnan(by.loc["Alpha", "CfgB"])
    assert math.isnan(by.loc["Alpha", "Delta (B-A)"])
    assert math.isnan(by.loc["Alpha", "% Change"])
    assert by.loc["Alpha", "Winner"] == ""
    # Beta only in B
    assert math.isnan(by.loc["Beta", "CfgA"])
    assert by.loc["Beta", "CfgB"] == 5.0
    assert by.loc["Beta", "Winner"] == ""
    # Sharpe higher better: A wins, delta -1, pct -0.5
    assert by.loc["Sharpe", "Delta (B-A)"] == pytest.approx(-1.0)
    assert by.loc["Sharpe", "% Change"] == pytest.approx(-0.5)
    assert by.loc["Sharpe", "Winner"] == "A"
    # Volatility lower better: A (0.1) wins despite B larger
    assert by.loc["Volatility", "Delta (B-A)"] == pytest.approx(0.1)
    assert by.loc["Volatility", "% Change"] == pytest.approx(1.0)
    assert by.loc["Volatility", "Winner"] == "A"
    # Zero A: pct_change stays None (guard abs(a)>1e-10), delta 0.5, winner B
    assert by.loc["Zero", "Delta (B-A)"] == pytest.approx(0.5)
    assert math.isnan(by.loc["Zero", "% Change"])
    assert by.loc["Zero", "Winner"] == "B"


def test_summary_df_lower_is_better_variants_and_tie() -> None:
    # max_drawdown (more negative is worse; lower value wins when lower_is_better)
    res_a = _result(user_stats={"max_drawdown": -0.05, "vol": 0.2, "sharpe": 1.0})
    res_b = _result(user_stats={"max_drawdown": -0.10, "vol": 0.2, "sharpe": 1.0})
    df = ce.build_comparison_summary_df(res_a, res_b)
    by = df.set_index("Metric")
    # -0.10 < -0.05, lower_is_better => B wins
    assert by.loc["max_drawdown", "Winner"] == "B"
    assert by.loc["vol", "Winner"] == "Tie"
    assert by.loc["sharpe", "Winner"] == "Tie"
    assert by.loc["sharpe", "Delta (B-A)"] == pytest.approx(0.0)
    assert by.loc["sharpe", "% Change"] == pytest.approx(0.0)


def test_summary_df_empty_when_no_metrics() -> None:
    df = ce.build_comparison_summary_df(_result(), _result())
    assert df.empty
    assert list(df.columns) == []


# ---------------------------------------------------------------- build_selection_differences_df


def test_selection_differences_exact() -> None:
    res_a = _result(period_results=[_period(funds=["F1", "F2"], weights={"F1": 0.6, "F2": 0.4})])
    res_b = _result(period_results=[_period(funds=["F2", "F3"], weights={"F2": 0.5, "F3": "0.5"})])
    df = ce.build_selection_differences_df(res_a, res_b, label_a="CfgA", label_b="CfgB")
    assert list(df.columns) == [
        "Period",
        "Fund",
        "In CfgA",
        "In CfgB",
        "CfgA Weight",
        "CfgB Weight",
        "Difference",
    ]
    assert len(df) == 2
    f1 = df[df["Fund"] == "F1"].iloc[0]
    assert f1["Period"] == "2024-01 → 2024-03"
    assert f1["In CfgA"] == "Yes"
    assert f1["In CfgB"] == "No"
    assert f1["CfgA Weight"] == pytest.approx(0.6)
    assert pd.isna(f1["CfgB Weight"])
    assert f1["Difference"] == "Only in A"
    f3 = df[df["Fund"] == "F3"].iloc[0]
    assert f3["In CfgA"] == "No"
    assert f3["In CfgB"] == "Yes"
    assert pd.isna(f3["CfgA Weight"])
    assert f3["CfgB Weight"] == pytest.approx(0.5)
    assert f3["Difference"] == "Only in B"
    assert "F2" not in set(df["Fund"])


def test_selection_differences_mismatched_period_counts_and_fallback_label() -> None:
    res_a = _result(
        period_results=[
            _period(funds=["F1"], weights={"F1": 1.0}),
            {"period": (), "selected_funds": ["FX"], "fund_weights": {"FX": 0.25}},
        ]
    )
    res_b = _result(period_results=[_period(funds=["F1"], weights={"F1": 1.0})])
    df = ce.build_selection_differences_df(res_a, res_b)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Period"] == "\u2192"
    assert row["Fund"] == "FX"
    assert row["Difference"] == "Only in A"
    assert row["Config A Weight"] == pytest.approx(0.25)


def test_selection_differences_identical_returns_empty() -> None:
    p = [_period(funds=["F1"], weights={"F1": 1.0})]
    df = ce.build_selection_differences_df(_result(period_results=p), _result(period_results=p))
    assert df.empty


def test_selection_differences_missing_keys() -> None:
    res_a = _result(period_results=[{}])
    res_b = _result(period_results=[{"selected_funds": ["F9"]}])
    df = ce.build_selection_differences_df(res_a, res_b)
    assert len(df) == 1
    assert df.iloc[0]["Fund"] == "F9"
    assert df.iloc[0]["Difference"] == "Only in B"
    assert df.iloc[0]["Period"] == "\u2192"


# ---------------------------------------------------------------- build_period_comparison_df


def test_period_comparison_exact_and_turnover_fallback() -> None:
    res_a = _result(
        period_results=[
            _period(funds=["F1", "F2"], turnover=0.30, transaction_cost=0.01),
            _period(
                start="2024-04",
                end="2024-06",
                funds=["F1"],
                turnover=None,
                risk_diagnostics={"turnover": "0.11"},
                transaction_cost="bad",
            ),
        ]
    )
    res_b = _result(
        period_results=[
            _period(
                funds=["F1", "F2", "F3"],
                risk_diagnostics={"turnover": 0.45},
                transaction_cost="0.02",
            ),
            _period(start="2024-04", end="2024-06", funds=[], transaction_cost=None),
        ]
    )
    df = ce.build_period_comparison_df(res_a, res_b, label_a="CfgA", label_b="CfgB")
    assert list(df.columns) == [
        "Period",
        "CfgA Funds Selected",
        "CfgB Funds Selected",
        "Δ Funds",
        "CfgA Turnover",
        "CfgB Turnover",
        "CfgA Txn Cost",
        "CfgB Txn Cost",
    ]
    assert df.iloc[0]["Period"] == "2024-01 → 2024-03"
    assert df.iloc[0]["CfgA Funds Selected"] == 2
    assert df.iloc[0]["CfgB Funds Selected"] == 3
    assert df.iloc[0]["Δ Funds"] == 1
    assert df.iloc[0]["CfgA Turnover"] == pytest.approx(0.30)
    assert df.iloc[0]["CfgB Turnover"] == pytest.approx(0.45)
    assert df.iloc[0]["CfgA Txn Cost"] == pytest.approx(0.01)
    assert df.iloc[0]["CfgB Txn Cost"] == pytest.approx(0.02)
    # Second row: A turnover via risk_diagnostics fallback, txn bad -> None
    assert df.iloc[1]["CfgA Turnover"] == pytest.approx(0.11)
    assert pd.isna(df.iloc[1]["CfgB Turnover"])
    assert pd.isna(df.iloc[1]["CfgA Txn Cost"])
    assert df.iloc[1]["Δ Funds"] == -1


def test_period_comparison_empty() -> None:
    df = ce.build_period_comparison_df(_result(period_results=[]), _result(period_results=[]))
    assert df.empty


def test_period_comparison_extra_period_in_b() -> None:
    res_a = _result(period_results=[])
    res_b = _result(period_results=[_period(funds=["F1"], turnover=0.2, transaction_cost=0.0)])
    df = ce.build_period_comparison_df(res_a, res_b)
    assert len(df) == 1
    assert df.iloc[0]["Config A Funds Selected"] == 0
    assert df.iloc[0]["Config B Funds Selected"] == 1
    assert df.iloc[0]["Δ Funds"] == 1
    assert pd.isna(df.iloc[0]["Config A Turnover"])
    assert df.iloc[0]["Config B Turnover"] == pytest.approx(0.2)


# ---------------------------------------------------------------- build_raw_comparison_df


def test_raw_comparison_with_score_frame() -> None:
    score = pd.DataFrame({"Score": [10.0, 20.0], "Rank": [1, 2]}, index=["F1", "F2"])
    res_a = _result(period_results=[_period(funds=["F1"], weights={"F1": 0.7}, score_frame=score)])
    res_b = _result(period_results=[_period(funds=["F2"], weights={"F2": 0.3}, score_frame=score)])
    df = ce.build_raw_comparison_df(res_a, res_b, label_a="CfgA", label_b="CfgB")
    assert list(df.columns) == [
        "Config",
        "Period",
        "Period_Num",
        "Fund",
        "Selected",
        "Weight",
        "InSample_Score",
        "InSample_Rank",
    ]
    assert len(df) == 4
    a_f1 = df[(df["Config"] == "CfgA") & (df["Fund"] == "F1")].iloc[0]
    assert a_f1["Selected"] == "Yes"
    assert a_f1["Weight"] == pytest.approx(0.7)
    assert a_f1["InSample_Score"] == pytest.approx(10.0)
    assert a_f1["Period_Num"] == 1
    assert a_f1["Period"] == "2024-01 → 2024-03"
    a_f2 = df[(df["Config"] == "CfgA") & (df["Fund"] == "F2")].iloc[0]
    assert a_f2["Selected"] == "No"
    assert pd.isna(a_f2["Weight"])
    assert a_f2["InSample_Score"] == pytest.approx(20.0)
    b_f2 = df[(df["Config"] == "CfgB") & (df["Fund"] == "F2")].iloc[0]
    assert b_f2["Selected"] == "Yes"
    assert b_f2["Weight"] == pytest.approx(0.3)


def test_raw_comparison_without_score_frame_uses_selected() -> None:
    res_a = _result(
        period_results=[{"period": (), "selected_funds": ["F9"], "fund_weights": {"F9": "0.5"}}]
    )
    res_b = _result(period_results=[])
    df = ce.build_raw_comparison_df(res_a, res_b)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Config"] == "Config A"
    assert row["Period"] == "\u2192"
    assert row["Fund"] == "F9"
    assert row["Selected"] == "Yes"
    assert row["Weight"] == pytest.approx(0.5)
    assert "InSample_Score" not in df.columns


def test_raw_comparison_empty_score_frame_falls_back() -> None:
    res_a = _result(
        period_results=[_period(funds=["F1"], weights={"F1": 1.0}, score_frame=pd.DataFrame())]
    )
    df = ce.build_raw_comparison_df(res_a, _result(period_results=[]))
    assert list(df["Fund"]) == ["F1"]


def test_raw_comparison_no_periods_empty() -> None:
    df = ce.build_raw_comparison_df(_result(period_results=[]), _result(period_results=[]))
    assert df.empty


# ---------------------------------------------------------------- by-period / by-fund summaries


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Config": "CfgA",
                "Period": "2024-01 → 2024-03",
                "Period_Num": 1,
                "Fund": "F1",
                "Selected": "Yes",
                "Weight": 0.6,
            },
            {
                "Config": "CfgA",
                "Period": "2024-01 → 2024-03",
                "Period_Num": 1,
                "Fund": "F2",
                "Selected": "No",
                "Weight": None,
            },
            {
                "Config": "CfgB",
                "Period": "2024-01 → 2024-03",
                "Period_Num": 1,
                "Fund": "F1",
                "Selected": "Yes",
                "Weight": 0.4,
            },
            {
                "Config": "CfgA",
                "Period": "2024-04 → 2024-06",
                "Period_Num": 2,
                "Fund": "F1",
                "Selected": "Yes",
                "Weight": 1.0,
            },
        ]
    )


def test_by_period_summary_exact() -> None:
    out = ce.build_by_period_summary_df(_raw_df())
    assert list(out.columns) == ["Period", "Period_Num", "Config", "Funds Selected", "Avg Weight"]
    assert len(out) == 3
    # Sorted by Period_Num, Config
    assert list(out["Config"]) == ["CfgA", "CfgB", "CfgA"]
    assert list(out["Funds Selected"]) == [1, 1, 1]
    assert out.iloc[0]["Avg Weight"] == pytest.approx(0.6)
    assert out.iloc[1]["Avg Weight"] == pytest.approx(0.4)
    assert out.iloc[2]["Avg Weight"] == pytest.approx(1.0)


def test_by_fund_summary_exact() -> None:
    out = ce.build_by_fund_summary_df(_raw_df())
    assert list(out.columns) == ["Fund", "Config", "Periods Selected", "Avg Weight"]
    # F1/CfgA selected twice avg (0.6+1.0)/2=0.8 ; F1/CfgB once 0.4 ; F2/CfgA 0, avg NaN
    f1a = out[(out["Fund"] == "F1") & (out["Config"] == "CfgA")].iloc[0]
    assert f1a["Periods Selected"] == 2
    assert f1a["Avg Weight"] == pytest.approx(0.8)
    f2a = out[(out["Fund"] == "F2") & (out["Config"] == "CfgA")].iloc[0]
    assert f2a["Periods Selected"] == 0
    assert pd.isna(f2a["Avg Weight"])


def test_by_summaries_empty_input() -> None:
    empty = pd.DataFrame()
    assert ce.build_by_period_summary_df(empty).empty
    assert ce.build_by_fund_summary_df(empty).empty


# ---------------------------------------------------------------- _write_dataframe_to_sheet


class _FakeWS:
    def __init__(self) -> None:
        self.cols: list[tuple[int, int, float]] = []
        self.writes: list[tuple[int, int, Any, Any]] = []
        self.frozen: tuple[int, int] | None = None
        self.filter: tuple[int, int, int, int] | None = None

    def set_column(self, a: int, b: int, w: float) -> None:
        self.cols.append((a, b, w))

    def write(self, r: int, c: int, v: Any, fmt: Any = None) -> None:
        self.writes.append((r, c, v, fmt))

    def freeze_panes(self, r: int, c: int) -> None:
        self.frozen = (r, c)

    def autofilter(self, a: int, b: int, c: int, d: int) -> None:
        self.filter = (a, b, c, d)


def test_write_dataframe_to_sheet_exact() -> None:
    ws = _FakeWS()
    df = pd.DataFrame([{"Name": "F1", "Val": 1.5}, {"Name": "F22", "Val": None}])
    ce._write_dataframe_to_sheet(ws, df, header_fmt="HF", workbook=object())
    # Headers at row 0 with HF
    assert (0, 0, "Name", "HF") in ws.writes
    assert (0, 1, "Val", "HF") in ws.writes
    assert (1, 0, "F1", None) in ws.writes
    assert (1, 1, 1.5, None) in ws.writes
    # None -> "" at row 2 col 1
    assert (2, 1, "", None) in ws.writes
    assert ws.frozen == (1, 0)
    assert ws.filter == (0, 0, 2, 1)
    assert len(ws.cols) == 2
    # Width: max(len(col), max str len)+2 capped 50: Name-> max(4,3)+2=6, Val->max(3,4)+2=6
    assert ws.cols[0] == (0, 0, 6)
    assert ws.cols[1] == (1, 1, 5)


def test_write_dataframe_to_sheet_empty_noop() -> None:
    ws = _FakeWS()
    ce._write_dataframe_to_sheet(ws, pd.DataFrame(), header_fmt="HF", workbook=object())
    assert ws.writes == []
    assert ws.cols == []
    assert ws.frozen is None


# ---------------------------------------------------------------- Excel workbook


def _full_results() -> tuple[SimpleNamespace, SimpleNamespace]:
    res_a = _result(
        user_stats={"Sharpe": 2.0, "CAGR": 0.10, "Zeta": 1.0},
        period_results=[
            _period(
                funds=["F1", "F2"],
                weights={"F1": 0.6, "F2": 0.4},
                turnover=0.3,
                transaction_cost=0.01,
            )
        ],
    )
    res_b = _result(
        user_stats={"Sharpe": 1.0, "CAGR": 0.12},
        period_results=[
            _period(
                funds=["F2", "F3"],
                weights={"F2": 0.5, "F3": 0.5},
                turnover=0.4,
                transaction_cost=0.02,
            )
        ],
    )
    return res_a, res_b


def test_excel_workbook_sheets_and_values() -> None:
    res_a, res_b = _full_results()
    data = ce.build_comparison_excel_workbook(
        res_a, res_b, {"lookback": 3, "name": "A"}, {"lookback": 6}, label_a="CfgA", label_b="CfgB"
    )
    assert isinstance(data, bytes)
    assert len(data) > 1000
    wb = openpyxl.load_workbook(BytesIO(data))
    assert wb.sheetnames == [
        "Comparison Summary",
        "Selection Differences",
        "Period Comparison",
        "Raw Data (Pivot-Ready)",
        "By Period",
        "By Fund",
        "CfgA Config",
        "CfgB Config",
    ]
    ws = wb["Comparison Summary"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, 7)]
    assert headers == ["Metric", "CfgA", "CfgB", "Delta (B-A)", "% Change", "Winner"]
    # Sorted metrics CAGR, Sharpe
    assert ws.cell(row=2, column=1).value == "CAGR"
    assert ws.cell(row=2, column=2).value == pytest.approx(0.10)
    assert ws.cell(row=2, column=3).value == pytest.approx(0.12)
    assert ws.cell(row=2, column=6).value == "B"
    assert ws.cell(row=3, column=1).value == "Sharpe"
    assert ws.cell(row=3, column=6).value == "A"
    # Config sheets contain JSON lines
    cfg_a_text = "\n".join(
        str(wb["CfgA Config"].cell(row=r, column=1).value or "") for r in range(1, 6)
    )
    assert '"lookback": 3' in cfg_a_text
    assert '"name": "A"' in cfg_a_text
    cfg_b_text = "\n".join(
        str(wb["CfgB Config"].cell(row=r, column=1).value or "") for r in range(1, 4)
    )
    assert '"lookback": 6' in cfg_b_text
    # Selection Differences has F1/F3 rows
    diff_ws = wb["Selection Differences"]
    diff_funds = {diff_ws.cell(row=r, column=2).value for r in range(2, 4)}
    assert diff_funds == {"F1", "F3"}


def test_excel_workbook_empty_results_minimal_sheets() -> None:
    data = ce.build_comparison_excel_workbook(
        _result(period_results=[]), _result(period_results=[]), {"a": 1}, {"b": 2}
    )
    wb = openpyxl.load_workbook(BytesIO(data))
    # No summary/period/raw/by sheets; diff sheet with message + 2 config sheets
    assert wb.sheetnames == ["Selection Differences", "Config A Config", "Config B Config"]
    assert wb["Selection Differences"].cell(row=1, column=1).value == (
        "No selection differences found between configurations."
    )


def test_excel_workbook_long_label_truncated_config_sheet() -> None:
    res_a, res_b = _full_results()
    long_a = "A" * 40
    data = ce.build_comparison_excel_workbook(res_a, res_b, {}, {}, label_a=long_a, label_b="B")
    wb = openpyxl.load_workbook(BytesIO(data))
    assert (long_a + " Config")[:31] in wb.sheetnames


# ---------------------------------------------------------------- CSV bundle


def test_csv_bundle_exact_contents() -> None:
    res_a, res_b = _full_results()
    data = ce.build_comparison_csv_bundle(
        res_a, res_b, {"lookback": 3}, {"lookback": 6}, label_a="CfgA", label_b="CfgB"
    )
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert names == {
            "comparison_summary.csv",
            "selection_differences.csv",
            "period_comparison.csv",
            "raw_data_pivot_ready.csv",
            "by_period_summary.csv",
            "by_fund_summary.csv",
            "config_CfgA.json",
            "config_CfgB.json",
        }
        assert json.loads(zf.read("config_CfgA.json")) == {"lookback": 3}
        assert json.loads(zf.read("config_CfgB.json")) == {"lookback": 6}
        summary = pd.read_csv(StringIO(zf.read("comparison_summary.csv").decode()))
        assert list(summary["Metric"]) == ["CAGR", "Sharpe", "Zeta"]
        sharpe = summary[summary["Metric"] == "Sharpe"].iloc[0]
        assert sharpe["CfgA"] == pytest.approx(2.0)
        assert sharpe["Winner"] == "A"
        diff = pd.read_csv(StringIO(zf.read("selection_differences.csv").decode()))
        assert set(diff["Fund"]) == {"F1", "F3"}
        period = pd.read_csv(StringIO(zf.read("period_comparison.csv").decode()))
        assert period.iloc[0]["CfgA Turnover"] == pytest.approx(0.3)
        raw = pd.read_csv(StringIO(zf.read("raw_data_pivot_ready.csv").decode()))
        assert set(raw["Config"]) == {"CfgA", "CfgB"}


def test_csv_bundle_empty_only_configs() -> None:
    data = ce.build_comparison_csv_bundle(
        _result(period_results=[]), _result(period_results=[]), {"x": 1}, {"y": 2}
    )
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert set(zf.namelist()) == {"config_Config_A.json", "config_Config_B.json"}
        assert json.loads(zf.read("config_Config_A.json")) == {"x": 1}


def test_csv_bundle_default_labels_config_names() -> None:
    res_a, res_b = _full_results()
    data = ce.build_comparison_csv_bundle(res_a, res_b, {}, {})
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert "config_Config_A.json" in zf.namelist()
        assert "config_Config_B.json" in zf.namelist()
