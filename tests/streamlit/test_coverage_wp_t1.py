"""Targeted tests for 3_Results.py covering product code work package t1."""

from __future__ import annotations

import importlib
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from tests.support.dummy_streamlit import DummyStreamlit


def _page() -> Any:
    return importlib.import_module("streamlit_app.pages.3_Results")


def test_fmt_pct_and_ratio_non_finite() -> None:
    page = _page()
    assert page._fmt_pct(float("nan")) == "—"
    assert page._fmt_pct(float("inf")) == "—"
    assert page._fmt_ratio(float("nan")) == "—"
    assert page._fmt_ratio(float("-inf")) == "—"


def test_should_auto_render_import_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page()
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", None)
    assert page._should_auto_render() is False


def test_analysis_error_messages_non_value_error() -> None:
    page = _page()
    err = RuntimeError("Database timeout during computation")
    summary, detail = page._analysis_error_messages(err)
    expected_summary = (
        "We ran into an unexpected problem while generating the report. "
        "Please try again or adjust your configuration."
    )
    assert summary == expected_summary
    assert detail == "Database timeout during computation"


def test_diagnostic_message_with_error_in_details() -> None:
    page = _page()
    res = SimpleNamespace(details={"error": "Optimization failure in sub-routine"})
    summary, detail = page._diagnostic_message(res)
    assert summary == "Analysis failed to produce results."
    assert detail == "Optimization failure in sub-routine"


def test_stats_value_from_series() -> None:
    page = _page()
    s = pd.Series({"Sharpe": 1.42, "Sortino": 2.15})
    assert page._stats_value(s, "sharpe") == 1.42
    assert page._stats_value(s, "sortino") == 2.15


def test_stats_value_from_object_and_unparseable() -> None:
    page = _page()
    obj_valid = SimpleNamespace(CAGR=0.18)
    assert page._stats_value(obj_valid, "cagr") == 0.18

    obj_bad = SimpleNamespace(cagr="not_a_number")
    assert page._stats_value(obj_bad, "cagr") is None


def test_coerce_weight_mapping() -> None:
    page = _page()
    weights = page._coerce_weight_mapping({"FundA": 2.0, "FundB": 2.0})
    assert weights == {"FundA": 0.5, "FundB": 0.5}


def test_compute_weighted_portfolio_returns_renormalisation() -> None:
    page = _page()
    # Empty inputs return empty series
    assert page._compute_weighted_portfolio_returns(pd.DataFrame(), {}).empty

    # Weights summing to 4.0 need renormalisation to sum to 1.0 (0.25 and 0.75)
    out_df = pd.DataFrame(
        {"FundA": [0.10, 0.20], "FundB": [0.02, 0.04]},
        index=["2020-01-31", "2020-02-29"],
    )
    weights = {"FundA": 1.0, "FundB": 3.0}
    # Expected: 0.10*0.25 + 0.02*0.75 = 0.040; 0.20*0.25 + 0.04*0.75 = 0.080
    res = page._compute_weighted_portfolio_returns(out_df, weights)
    assert res.name == "Portfolio"
    assert res.tolist() == [0.04, 0.08]


def test_portfolio_series_from_details_multi_period() -> None:
    page = _page()
    details = {
        "period_results": [
            "invalid_non_dict_entry",
            {"out_sample_scaled": pd.DataFrame()},
            {
                "out_sample_scaled": pd.DataFrame(
                    {"FundA": [0.04]},
                    index=["2020-01-31"],
                ),
                "fund_weights": {"FundA": 1.0},
            },
            {
                "out_sample_scaled": pd.DataFrame(
                    {"FundA": [0.99, 0.08]},
                    index=["2020-01-31", "2020-02-29"],
                ),
                "ew_weights": {"FundA": 1.0},
            },
        ]
    }
    series = page._portfolio_series_from_details(details)
    # 2020-01-31 kept from first period (0.04), 2020-02-29 kept from second period (0.08)
    assert series.index.tolist() == ["2020-01-31", "2020-02-29"]
    assert series.tolist() == [0.04, 0.08]


def test_portfolio_series_from_details_single_period_and_empty() -> None:
    page = _page()
    details = {
        "out_sample_scaled": pd.DataFrame({"FundA": [0.05]}, index=["2020-01-31"]),
        "fund_weights": {"FundA": 1.0},
    }
    res = page._portfolio_series_from_details(details)
    assert res.tolist() == [0.05]

    empty_res = page._portfolio_series_from_details({})
    assert empty_res.empty


def test_result_period_count_invalid_count_exception() -> None:
    page = _page()
    res = SimpleNamespace(period_count="not_an_int")
    details = {"period_results": [{"p": 1}, {"p": 2}]}
    assert page._result_period_count(res, details) == 2


def test_comparison_cache_reset_when_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page()
    monkeypatch.setattr(page.st, "session_state", {"comparison_results_cache": "corrupted_string"})
    cache = page._comparison_cache()
    assert cache == {}
    assert page.st.session_state["comparison_results_cache"] == {}
    assert cache is page.st.session_state["comparison_results_cache"]


def test_rolling_sharpe_empty() -> None:
    page = _page()
    empty_series = pd.Series([], dtype=float)
    assert page._rolling_sharpe(empty_series).empty


def test_get_portfolio_funds_by_period_weights_and_selected() -> None:
    page = _page()
    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2020-01-01", "2020-03-31", "2020-04-01", "2020-06-30"),
                    "fund_weights": {"FundA": 0.5, "FundZero": 0.0, "FundBad": "not_numeric"},
                },
                {
                    "period": ("2020-04-01", "2020-06-30", "2020-07-01", "2020-09-30"),
                    "fund_weights": {},
                    "selected_funds": ["FundB", "FundC", ""],
                },
            ]
        }
    )
    by_period = page._get_portfolio_funds_by_period(res)
    assert by_period["2020-04-01"] == {"FundA"}
    assert by_period["2020-07-01"] == {"FundB", "FundC"}


def test_get_final_portfolio() -> None:
    page = _page()
    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2020-01-01", "2020-03-31", "2020-04-01", "2020-06-30"),
                    "fund_weights": {"FundA": 1.0},
                },
                {
                    "period": ("2020-04-01", "2020-06-30", "2020-07-01", "2020-09-30"),
                    "fund_weights": {"FundB": 0.5, "FundC": 0.5},
                },
            ]
        }
    )
    assert page._get_final_portfolio(res) == {"FundB", "FundC"}
    assert page._get_final_portfolio(SimpleNamespace(details={})) == set()


def test_compute_trailing_stats_empty() -> None:
    page = _page()
    empty = pd.Series([], dtype=float)
    assert page._compute_trailing_stats(empty).empty


def test_format_change_reason_all_branches() -> None:
    page = _page()
    r1 = page._format_change_reason("z_entry", "zscore=1.456 extra", "added")
    assert r1 == "Performance above threshold (z=1.46)"

    r2 = page._format_change_reason("z_exit", "zscore=-2.103 extra", "dropped")
    assert r2 == "Performance below threshold (z=-2.10)"

    r3 = page._format_change_reason("z_entry", "zscore=invalid", "added")
    assert r3 == "Performance above threshold"

    r4 = page._format_change_reason("custom_momentum_rule", "", "added")
    assert r4 == "Custom Momentum Rule"


def test_extract_manager_changes_initial_and_transitions() -> None:
    page = _page()
    assert page._extract_manager_changes(SimpleNamespace(details={})).empty

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 0.5, "FundB": 0.5, "FundBad": "not_numeric"},
                    "manager_changes": [],
                },
                {
                    "period": ("2020-01-01", "2020-03-31", "2020-04-01", "2020-06-30"),
                    "fund_weights": {"FundB": 0.5, "FundC": 0.5},
                    "manager_changes": [
                        {
                            "manager": "FundC",
                            "action": "added",
                            "reason": "z_entry",
                            "detail": "zscore=2.10",
                        },
                        {
                            "manager": "FundA",
                            "action": "dropped",
                            "reason": "z_exit",
                            "detail": "zscore=-1.50",
                        },
                    ],
                },
                {
                    # Period 3: fund_weights empty, relies on selected_funds, FundD added without event
                    "period": ("2020-04-01", "2020-06-30", "2020-07-01", "2020-09-30"),
                    "fund_weights": {},
                    "selected_funds": ["FundB", "FundD"],
                    "manager_changes": [],
                },
            ]
        }
    )
    changes = page._extract_manager_changes(res)
    assert len(changes) == 6
    # Initial records
    assert changes.iloc[0]["Action"] == "Initial"
    assert changes.iloc[0]["Manager"] == "FundA"
    assert changes.iloc[0]["Reason"] == "Initial portfolio selection"
    assert changes.iloc[1]["Action"] == "Initial"
    assert changes.iloc[1]["Manager"] == "FundB"

    # Transitions period 2
    hired = changes[changes["Action"] == "Hired"].iloc[0]
    assert hired["Manager"] == "FundC"
    assert hired["Reason"] == "Performance above threshold (z=2.10)"

    terminated = changes[changes["Action"] == "Terminated"].iloc[0]
    assert terminated["Manager"] == "FundA"
    assert terminated["Reason"] == "Performance below threshold (z=-1.50)"

    # Transitions period 3: FundD added without event record
    hired_d = changes[changes["Manager"] == "FundD"].iloc[0]
    assert hired_d["Action"] == "Hired"
    assert hired_d["Reason"] == ""


def test_extract_manager_decisions_all_actions() -> None:
    page = _page()
    assert page._extract_manager_decisions(SimpleNamespace(details={})).empty

    # Subcase with no valid added/dropped/skipped actions returns empty DataFrame
    res_none = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "manager_changes": [{"manager": "FundX", "action": "ignored"}],
                }
            ]
        }
    )
    assert page._extract_manager_decisions(res_none).empty

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "manager_changes": [
                        {
                            "manager": "FundA",
                            "action": "added",
                            "firm": "FirmA",
                            "reason": "seed",
                            "detail": "",
                        },
                        {
                            "manager": "FundB",
                            "action": "dropped",
                            "firm": "FirmB",
                            "reason": "z_exit",
                            "detail": "zscore=-2.00",
                        },
                        {
                            "manager": "FundC",
                            "action": "skipped",
                            "firm": "FirmC",
                            "reason": "one_per_firm",
                            "detail": "",
                        },
                        {
                            "manager": "FundD",
                            "action": "rebalance",
                            "firm": "FirmD",
                            "reason": "ignored",
                            "detail": "",
                        },
                    ],
                }
            ]
        }
    )
    df = page._extract_manager_decisions(res)
    assert len(df) == 3
    assert df["Action"].tolist() == ["Hired", "Terminated", "Skipped"]
    assert df["Manager"].tolist() == ["FundA", "FundB", "FundC"]
    assert df["Firm"].tolist() == ["FirmA", "FirmB", "FirmC"]
    assert df["Reason"].tolist() == [
        "Initial seed",
        "Performance below threshold (z=-2.00)",
        "Firm concentration limit",
    ]


def test_compute_fund_stats_calculations_and_short_series() -> None:
    page = _page()
    assert page._compute_fund_stats(pd.Series([], dtype=float)) == {}
    assert page._compute_fund_stats(pd.Series([0.05])) == {}

    s = pd.Series([0.02, 0.04, -0.01, -0.03])
    stats = page._compute_fund_stats(s)
    assert stats["CAGR"] == pytest.approx(0.057113, abs=1e-5)
    assert stats["Vol"] == pytest.approx(0.107703, abs=1e-5)
    assert stats["Sharpe"] == pytest.approx(0.530279, abs=1e-5)
    assert stats["Sortino"] == pytest.approx(1.165810, abs=1e-5)
    assert stats["IR"] == pytest.approx(0.557086, abs=1e-5)
    assert stats["Max DD"] == pytest.approx(-0.039700, abs=1e-5)


def test_compute_fund_holding_periods_seed_and_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page()
    monkeypatch.setattr(page.st, "session_state", {"model_state": {"multi_period_frequency": "Q"}})
    empty_summary, empty_risk = page._compute_fund_holding_periods(SimpleNamespace(details={}))
    assert empty_summary.empty
    assert empty_risk.empty

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "manager_changes": [
                        {"manager": "FundA", "action": "added", "reason": "seed"},
                    ],
                    "out_sample_scaled": pd.DataFrame({"FundA": [0.02, 0.04]}),
                },
                {
                    "period": ("2020-01-01", "2020-03-31", "2020-04-01", "2020-06-30"),
                    "manager_changes": [
                        {"manager": "FundB", "action": "added", "reason": "z_entry"},
                        {"manager": "FundA", "action": "dropped", "reason": "z_exit"},
                    ],
                    "out_sample_scaled": pd.DataFrame(
                        {"FundA": [-0.01, -0.03], "FundB": [0.05, 0.02]}
                    ),
                },
            ]
        }
    )
    summary_df, risk_df = page._compute_fund_holding_periods(res)
    assert len(summary_df) == 2
    # FundA was seed: entry is simulation start date 2020-01-01, exit 2020-04-01, 2 quarters = 0.5 years
    row_a = summary_df[summary_df["Manager"] == "FundA"].iloc[0]
    assert row_a["Entry"] == "2020-01-01"
    assert row_a["Exit"] == "2020-04-01"
    assert row_a["Years Held"] == 0.5

    # FundB entered 2020-04-01, exit Current, 1 quarter = 0.25 years
    row_b = summary_df[summary_df["Manager"] == "FundB"].iloc[0]
    assert row_b["Entry"] == "2020-04-01"
    assert row_b["Exit"] == "Current"
    assert row_b["Years Held"] == 0.25

    # Risk stats computed from accumulated returns
    assert len(risk_df) == 2
    assert "CAGR" in risk_df.columns
    assert "Vol" in risk_df.columns


def test_build_period_detail_defaults_and_explicit() -> None:
    page = _page()
    raw = {
        "period": ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        "score_frame": None,
        "selected_funds": ["FundA"],
        "fund_weights": {"FundA": 1.0},
        "ew_weights": {"FundA": 1.0},
        "in_sample_stats": {"CAGR": 0.10},
        "out_sample_stats": {"CAGR": 0.08},
        "manager_changes": [],
        "out_sample_scaled": None,
        "out_user_stats": {"Sharpe": 1.2},
        "out_ew_stats": {"Sharpe": 1.0},
    }
    detail = page._build_period_detail(raw, period_num=5)
    assert detail["period_num"] == 5
    assert detail["in_start"] == "2019-01-01"
    assert detail["in_end"] == "2019-12-31"
    assert detail["out_start"] == "2020-01-01"
    assert detail["out_end"] == "2020-12-31"
    assert isinstance(detail["score_frame"], pd.DataFrame)
    assert detail["score_frame"].empty
    assert isinstance(detail["out_sample_scaled"], pd.DataFrame)
    assert detail["out_sample_scaled"].empty
    assert detail["selected_funds"] == ["FundA"]


def test_build_period_weights_df_with_rebalance_weights() -> None:
    page = _page()
    reb_weights = pd.DataFrame(
        {"FundA": [0.60, 0.00], "FundB": [0.40, "invalid_nan"]},
        index=["2020-01-15", "2020-02-15"],
    )
    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "rebalance_weights": reb_weights,
                }
            ]
        }
    )
    df = page._build_period_weights_df(res)
    assert len(df) == 2
    assert df["Period Start"].tolist() == ["2020-01-15", "2020-01-15"]
    assert df["Fund"].tolist() == ["FundA", "FundB"]
    assert df["Weight"].tolist() == [0.60, 0.40]


def test_build_period_weights_df_fallback_weights() -> None:
    page = _page()
    assert page._build_period_weights_df(SimpleNamespace(details={})).empty

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 0.70, "FundB": 0.30},
                }
            ]
        }
    )
    df = page._build_period_weights_df(res)
    assert len(df) == 2
    fund_to_weight = dict(zip(df["Fund"], df["Weight"], strict=False))
    assert fund_to_weight == {"FundA": 0.70, "FundB": 0.30}


def test_build_period_returns_df() -> None:
    page = _page()
    assert page._build_period_returns_df(SimpleNamespace(details={})).empty

    # Period results with empty out_sample_scaled returns empty DataFrame
    res_empty_oos = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 1.0},
                    "out_sample_scaled": pd.DataFrame(),
                }
            ]
        }
    )
    assert page._build_period_returns_df(res_empty_oos).empty

    out_df = pd.DataFrame(
        {"FundA": [0.01, 0.02], "FundUnselected": [0.99, 0.99]},
        index=["2020-01-31", "2020-02-29"],
    )
    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 1.0},
                    "out_sample_scaled": out_df,
                }
            ]
        }
    )
    df = page._build_period_returns_df(res)
    assert "FundUnselected" not in df.columns
    assert "FundA" in df.columns
    assert df["FundA"].tolist() == [0.01, 0.02]
    assert df["Period Start"].tolist() == ["2020-01-01", "2020-01-01"]
    assert df["Period End"].tolist() == ["2020-03-31", "2020-03-31"]


def test_format_weights_for_csv() -> None:
    page = _page()
    assert page._format_weights_for_csv(pd.DataFrame()).empty

    df = pd.DataFrame({"Fund": ["FundA", "FundB"], "Weight": [0.25, float("nan")]})
    formatted = page._format_weights_for_csv(df)
    assert formatted["Weight"].tolist() == ["25.0%", ""]


def test_format_returns_for_csv() -> None:
    page = _page()
    assert page._format_returns_for_csv(pd.DataFrame()).empty

    df = pd.DataFrame(
        {
            "Period Start": ["2020-01-01"],
            "Date": ["2020-01-31"],
            "FundA": [0.0125],
            "FundB": [float("inf")],
        }
    )
    formatted = page._format_returns_for_csv(df)
    assert formatted["Period Start"].iloc[0] == "2020-01-01"
    assert formatted["Date"].iloc[0] == "2020-01-31"
    assert formatted["FundA"].iloc[0] == "1.25%"
    assert formatted["FundB"].iloc[0] == ""


def test_format_holdings_for_csv() -> None:
    page = _page()
    assert page._format_holdings_for_csv(pd.DataFrame()).empty

    df = pd.DataFrame(
        {
            "CAGR": [0.125],
            "Vol": [0.082],
            "Max DD": [-0.054],
            "Sharpe": [1.456],
            "Sortino": [2.102],
            "IR": [0.893],
            "Years Held": [2.54],
            "Years": [3.41],
        }
    )
    formatted = page._format_holdings_for_csv(df)
    assert formatted["CAGR"].iloc[0] == "12.5%"
    assert formatted["Vol"].iloc[0] == "8.2%"
    assert formatted["Max DD"].iloc[0] == "-5.4%"
    assert formatted["Sharpe"].iloc[0] == "1.46"
    assert formatted["Sortino"].iloc[0] == "2.10"
    assert formatted["IR"].iloc[0] == "0.89"
    assert formatted["Years Held"].iloc[0] == "2.5"
    assert formatted["Years"].iloc[0] == "3.4"


def test_get_current_exposures_branches() -> None:
    page = _page()
    # Case 1: no period_results, extracted from analysis.exposures
    exp = pd.Series({"FundA": 0.6, "FundB": 0.4})
    res_analysis = SimpleNamespace(analysis=SimpleNamespace(exposures=exp), details={})
    assert page._get_current_exposures(res_analysis) is exp

    # Case 2: period_results present but empty weights
    res_no_weights = SimpleNamespace(details={"period_results": [{}]})
    assert page._get_current_exposures(res_no_weights) is None


def test_render_download_section_xlsx_export_and_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _page()
    stub = DummyStreamlit()
    downloads: list[dict[str, Any]] = []

    def custom_download_button(*args: Any, **kwargs: Any) -> None:
        downloads.append({"args": args, "kwargs": kwargs})

    stub.download_button = custom_download_button
    for attr in ["columns", "divider", "subheader", "caption"]:
        monkeypatch.setattr(page.st, attr, getattr(stub, attr))
    monkeypatch.setattr(page.st, "download_button", custom_download_button)
    monkeypatch.setattr(page.st, "session_state", {})

    # Mock export_phase1_workbook to write test bytes
    def mock_export(period_results: Any, path: str) -> None:
        Path(path).write_bytes(b"MOCK_PHASE1_XLSX")

    monkeypatch.setattr("trend_analysis.export.export_phase1_workbook", mock_export)
    # Mock flat_frames_from_results to return empty dict to test line 1901
    monkeypatch.setattr("trend_analysis.export.flat_frames_from_results", lambda res: {})

    # Trigger Path.unlink failure to cover cleanup exception handler
    orig_unlink = Path.unlink

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".xlsx":
            raise OSError("Simulated unlink failure")
        orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 1.0},
                    "out_sample_scaled": pd.DataFrame({"FundA": [0.02]}),
                }
            ]
        }
    )
    page._render_download_section(res, include_narrative=False)

    xlsx_dl = [d for d in downloads if "XLSX" in str(d["args"]) or "XLSX" in str(d["kwargs"])]
    assert len(xlsx_dl) == 1
    assert xlsx_dl[0]["kwargs"]["data"] == b"MOCK_PHASE1_XLSX"


def test_render_download_section_csv_zip_and_format_phase1_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _page()
    stub = DummyStreamlit()
    downloads: list[dict[str, Any]] = []

    def custom_download_button(*args: Any, **kwargs: Any) -> None:
        downloads.append({"args": args, "kwargs": kwargs})

    stub.download_button = custom_download_button
    for attr in ["columns", "divider", "subheader", "caption"]:
        monkeypatch.setattr(page.st, attr, getattr(stub, attr))
    monkeypatch.setattr(page.st, "download_button", custom_download_button)
    monkeypatch.setattr(page.st, "session_state", {})

    def mock_export(period_results: Any, path: str) -> None:
        Path(path).write_bytes(b"dummy")

    monkeypatch.setattr("trend_analysis.export.export_phase1_workbook", mock_export)

    fake_frames = {
        "summary": pd.DataFrame(
            {
                "OS Total Return": [15.5],
                "Weight": [25.0],
                "Contribution Share": [0.35],
                "OS Sharpe": [1.456],
                "Benchmark IR": [0.852],
                "Years": [4.2],
                "z_score": [1.987],
                "blended": [0.765],
                "OS Months": [24.0],
            }
        ),
        "periods": pd.DataFrame(
            {
                "CAGR": [10.2],
                "Vol": [5.1],
                "MaxDD": [-8.3],
                "Total Return": [12.0],
            }
        ),
        "empty_sheet": pd.DataFrame(),
    }
    monkeypatch.setattr("trend_analysis.export.flat_frames_from_results", lambda res: fake_frames)

    res = SimpleNamespace(
        details={
            "period_results": [
                {
                    "period": ("2019-10-01", "2019-12-31", "2020-01-01", "2020-03-31"),
                    "fund_weights": {"FundA": 1.0},
                    "out_sample_scaled": pd.DataFrame({"FundA": [0.02]}),
                }
            ]
        }
    )
    page._render_download_section(res, include_narrative=True)

    zip_dl = [d for d in downloads if "ZIP" in str(d["args"]) or "ZIP" in str(d["kwargs"])]
    assert len(zip_dl) == 1
    zip_bytes = zip_dl[0]["kwargs"]["data"]
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    assert "trend_phase1_summary.csv" in names
    assert "trend_phase1_periods.csv" in names
    assert "trend_phase1_empty_sheet.csv" not in names

    summary_csv = zf.read("trend_phase1_summary.csv").decode("utf-8")
    summary_df = pd.read_csv(io.StringIO(summary_csv), dtype=str)
    assert summary_df["OS Total Return"].iloc[0] == "15.5%"
    assert summary_df["Weight"].iloc[0] == "25.0%"
    assert summary_df["Contribution Share"].iloc[0] == "35.0%"
    assert summary_df["OS Sharpe"].iloc[0] == "1.46"
    assert summary_df["Benchmark IR"].iloc[0] == "0.85"
    assert summary_df["Years"].iloc[0] == "4.2"
    assert summary_df["z_score"].iloc[0] == "1.99"
    assert summary_df["blended"].iloc[0] == "0.77"
    assert summary_df["OS Months"].iloc[0] == "24"

    periods_csv = zf.read("trend_phase1_periods.csv").decode("utf-8")
    periods_df = pd.read_csv(io.StringIO(periods_csv), dtype=str)
    assert periods_df["CAGR"].iloc[0] == "10.2%"
    assert periods_df["Vol"].iloc[0] == "5.1%"
    assert periods_df["MaxDD"].iloc[0] == "-8.3%"
    assert periods_df["Total Return"].iloc[0] == "12.0%"
