"""Rendering tests for the Results page (``streamlit_app/pages/3_Results.py``).

The page is driven through its own functions with the shared ``DummyStreamlit`` stub (a thin
subclass adds the two widget signatures the page needs that the shared stub drops) and small
hand-built result objects. Every test asserts what the page rendered: exact captions, markdown,
metrics, dataframes (including the colours ``Styler.apply`` paints), messages and the arguments
handed to the download buttons, charts and LLM panels.

The LLM surfaces (``explain_results`` / ``comparison_llm``) and the Excel export are replaced by
recording stubs, so nothing here reaches a provider, the network or a subprocess.

Period Return tests cover the engine's ``max_drawdown`` field and legacy dictionary aliases.
The manager timeline regression asserts that decision-log events render even when there are
no holdings.
"""

from __future__ import annotations

import ast
import importlib
import io
import re
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler

from tests.support.dummy_streamlit import DummyStreamlit

RESULTS_PAGE = "streamlit_app.pages.3_Results"


class PageStub(DummyStreamlit):
    """``DummyStreamlit`` plus the widget signatures the Results page relies on.

    The page calls ``st.download_button(label, data, file_name, mime)`` positionally and passes a
    ``format_func`` to ``st.selectbox``; the shared stub accepts keywords only for the former and
    drops the latter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.selectbox_format_labels: list[list[str]] = []
        self.selectbox_keys: list[str | None] = []

    def download_button(
        self,
        label: str = "",
        data: Any = None,
        file_name: str | None = None,
        mime: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.downloads.append(
            {"label": label, "data": data, "file_name": file_name, "mime": mime, **kwargs}
        )

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        self.selectbox_keys.append(kwargs.get("key"))
        format_func = kwargs.get("format_func")
        if format_func is not None:
            self.selectbox_format_labels.append([format_func(o) for o in kwargs["options"]])
        return super().selectbox(*args, **kwargs)


@pytest.fixture()
def results_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, PageStub]:
    """The Results page module with ``st`` replaced by a fresh ``PageStub``.

    ``presentation_safe`` is the default profile so LLM surfaces stay off unless a test opts in.
    """
    monkeypatch.setenv("TREND_DEMO_PROFILE", "presentation_safe")
    stub = PageStub()
    page = importlib.import_module(RESULTS_PAGE)
    monkeypatch.setattr(page, "st", stub)
    monkeypatch.setattr(page.app_state, "st", stub)
    return page, stub


def _frame_of(shown: Any) -> pd.DataFrame:
    """The DataFrame behind an object handed to ``st.dataframe`` (unwraps a ``Styler``)."""
    return shown.data if isinstance(shown, Styler) else shown


def _painted_rows(shown: Styler) -> dict[int, str]:
    """Row position -> the one background colour ``Styler.apply`` painted over that whole row."""
    html = shown.to_html()
    cells: dict[tuple[int, int], str] = {}
    for selectors, colour in re.findall(r"([^{}]*)\{\s*background-color:\s*(#[0-9a-f]{6});", html):
        for row, col in re.findall(r"_row(\d+)_col(\d+)", selectors):
            cells[(int(row), int(col))] = colour
    painted: dict[int, str] = {}
    for row in sorted({row for row, _col in cells}):
        colours = {cells.get((row, col)) for col in range(len(shown.data.columns))}
        assert len(colours) == 1 and None not in colours, f"row {row} is only partly painted"
        painted[row] = colours.pop()
    return painted


def _period(out_start: str, out_end: str, **fields: Any) -> dict[str, Any]:
    """One ``period_results`` entry; the in-sample window is fixed and irrelevant to the tests."""
    return {"period": ("2019-01-31", "2019-12-31", out_start, out_end), **fields}


def _year_of_returns() -> pd.Series:
    """Twelve months alternating +3% / -1%: total return (1.03 * 0.99) ** 6 - 1 = 12.4%."""
    return pd.Series([0.03, -0.01] * 6, index=pd.date_range("2020-01-31", periods=12, freq="ME"))


# ---------------------------------------------------------------------------
# Comparison results
# ---------------------------------------------------------------------------


def test_comparison_overview_tabulates_deltas_and_bundles_them(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    monkeypatch.setattr(
        page.comparison_llm,
        "render_comparison_llm",
        lambda **_kwargs: pytest.fail("the LLM tab must not render in presentation_safe"),
    )
    result_a = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.0], "CAGR": [0.25]}),
        details={
            "period_results": [
                _period(
                    "2020-04-30",
                    "2020-06-30",
                    selected_funds=["Alpha", "Beta"],
                    turnover=0.5,
                    transaction_cost=0.25,
                    manager_changes=[{"manager": "Alpha", "action": "added", "reason": "seed"}],
                )
            ]
        },
    )
    result_b = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.5], "CAGR": [0.5]}),
        details={
            "period_results": [
                _period(
                    "2020-04-30",
                    "2020-06-30",
                    selected_funds=["Alpha"],
                    turnover=1.0,
                    transaction_cost=0.5,
                    manager_changes=[
                        {"manager": "Alpha", "action": "added", "reason": "seed"},
                        {"manager": "Beta", "action": "dropped", "reason": "z_exit"},
                    ],
                )
            ]
        },
    )
    model_a = {"lag": 1, "window": 63}
    model_b = {"lag": 1, "window": 126}

    page._render_comparison_results(
        "Config A", "Config B", result_a, result_b, model_a, model_b, data_hash="h1"
    )

    assert stub.tab_groups == [["Overview"]]
    assert stub.subheaders == [
        "Top-level metrics",
        "Multi-period summary",
        "Manager change counts by reason",
    ]
    assert stub.captions == [
        "Runs are cached by dataset fingerprint, selected funds, benchmark, and configuration.",
        "Bundle includes configs A/B, config diff text, and comparison CSVs for metrics, "
        "periods, and manager changes.",
    ]
    metrics, periods, managers = stub.dataframes
    assert metrics.to_dict("records") == [
        {"Metric": "CAGR", "Config A": 0.25, "Config B": 0.5, "Delta (B - A)": 0.25},
        {"Metric": "Sharpe", "Config A": 1.0, "Config B": 1.5, "Delta (B - A)": 0.5},
    ]
    assert periods.to_dict("records") == [
        {
            "Period": "2020-04-30 → 2020-06-30",
            "Config A Selected Funds": 2,
            "Config B Selected Funds": 1,
            "Selected Funds Δ (B - A)": -1,
            "Config A Turnover": 0.5,
            "Config B Turnover": 1.0,
            "Turnover Δ (B - A)": 0.5,
            "Config A Transaction Cost": 0.25,
            "Config B Transaction Cost": 0.5,
            "Transaction Cost Δ (B - A)": 0.25,
        }
    ]
    assert managers["Reason"].tolist() == ["seed", "z_exit"]
    assert managers["Config B Count"].tolist() == [1, 1]
    assert len(stub.downloads) == 1
    download = stub.downloads[0]
    assert {key: value for key, value in download.items() if key != "data"} == {
        "label": "⬇️ Export comparison bundle",
        "file_name": "comparison_bundle.zip",
        "mime": "application/zip",
    }
    bundle = zipfile.ZipFile(io.BytesIO(download["data"]))
    assert bundle.namelist() == [
        "config_A.json",
        "config_B.json",
        "config_diff.txt",
        "metrics_compare.csv",
        "periods_compare.csv",
        "manager_change_compare.csv",
    ]
    assert bundle.read("config_A.json").decode() == '{\n  "lag": 1,\n  "window": 63\n}'
    assert bundle.read("config_B.json").decode() == '{\n  "lag": 1,\n  "window": 126\n}'
    assert bundle.read("config_diff.txt").decode() == "~ window: (Config A) 63 -> (Config B) 126"
    assert bundle.read("metrics_compare.csv").decode() == metrics.to_csv(index=False)
    assert bundle.read("periods_compare.csv").decode() == periods.to_csv(index=False)
    assert bundle.read("manager_change_compare.csv").decode() == managers.to_csv(index=False)


def test_comparison_without_metrics_or_periods_shows_placeholders(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    empty = SimpleNamespace(metrics=None, details={})

    page._render_comparison_results(
        "Config A", "Config B", empty, empty, {"window": 63}, {"window": 63}, data_hash="h1"
    )

    assert stub.captions == [
        "Runs are cached by dataset fingerprint, selected funds, benchmark, and configuration.",
        "No comparable metrics available.",
        "No period-level results available to compare.",
        "No manager changes recorded in either run.",
        "Bundle includes configs A/B, config diff text, and comparison CSVs for metrics, "
        "periods, and manager changes.",
    ]
    assert stub.dataframes == []
    bundle = zipfile.ZipFile(io.BytesIO(stub.downloads[0]["data"]))
    assert bundle.namelist() == ["config_A.json", "config_B.json", "config_diff.txt"]
    assert bundle.read("config_diff.txt").decode() == "No differences found."


def test_comparison_llm_tab_gets_labels_diff_and_a_run_key_that_tracks_the_inputs(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    monkeypatch.setenv("TREND_DEMO_PROFILE", "public_llm_demo")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        page.comparison_llm, "render_comparison_llm", lambda **kwargs: calls.append(kwargs)
    )
    result_a = SimpleNamespace(metrics=None, details={})
    result_b = SimpleNamespace(metrics=None, details={})
    model_a = {"lag": 1, "window": 63}
    model_b = {"lag": 1, "window": 126}

    def render(
        state_b: dict[str, Any], data_hash: str, label_b: str = "Config B"
    ) -> dict[str, Any]:
        page._render_comparison_results(
            "Config A", label_b, result_a, result_b, model_a, state_b, data_hash=data_hash
        )
        return calls[-1]

    first = render(model_b, "h1")

    assert len(calls) == 1
    assert stub.tab_groups == [["Overview", "LLM Analysis"]]
    assert first["result_a"] is result_a
    assert first["result_b"] is result_b
    assert first["label_a"] == "Config A"
    assert first["label_b"] == "Config B"
    assert first["config_diff"] == "~ window: (Config A) 63 -> (Config B) 126"
    key = first["run_key"]
    assert re.fullmatch(r"[0-9a-f]{16}", key)
    assert render(model_b, "h1")["run_key"] == key
    assert render(model_b, "h2")["run_key"] != key
    assert render({"lag": 1, "window": 252}, "h1")["run_key"] != key
    assert render(model_b, "h1", label_b="Config C")["run_key"] != key


# ---------------------------------------------------------------------------
# Trailing statistics and the summary block
# ---------------------------------------------------------------------------


def test_trailing_stats_without_returns_show_a_placeholder_caption(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page

    page._render_trailing_stats(pd.Series(dtype=float))

    assert stub.captions == ["Insufficient data for trailing statistics."]
    assert stub.dataframes == []

    # Opposite state: a full year gets the formatted table and no new caption.
    page._render_trailing_stats(_year_of_returns())

    assert stub.captions == ["Insufficient data for trailing statistics."]
    assert [frame.to_dict("index") for frame in stub.dataframes] == [
        {
            "1Y": {
                "CAGR": "12.4%",
                "Volatility": "7.2%",
                "Sharpe": "1.72",
                "Sortino": "0.00",
                "Max DD": "-1.0%",
            },
            "LOF": {
                "CAGR": "12.4%",
                "Volatility": "7.2%",
                "Sharpe": "1.72",
                "Sortino": "0.00",
                "Max DD": "-1.0%",
            },
        }
    ]


def test_summary_prefers_trailing_stats_then_metrics_then_a_placeholder(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    metrics = pd.DataFrame({"Sharpe": [1.0]})

    page._render_summary(SimpleNamespace(metrics=metrics), _year_of_returns())

    assert stub.subheaders == ["📊 Portfolio Performance Summary"]
    assert [frame.index.tolist() for frame in stub.dataframes] == [["1Y", "LOF"]]
    assert stub.captions == []

    page._render_summary(SimpleNamespace(metrics=metrics), None)

    assert len(stub.dataframes) == 2
    assert stub.dataframes[1] is metrics
    assert stub.captions == []

    page._render_summary(SimpleNamespace(metrics=pd.DataFrame()), pd.Series(dtype=float))

    assert len(stub.dataframes) == 2
    assert stub.captions == ["No summary metrics available."]
    assert stub.subheaders == ["📊 Portfolio Performance Summary"] * 3


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------


def test_selection_criteria_lists_the_minimum_only_when_one_is_set(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    stub.session_state["model_state"] = {
        "selection_count": 6,
        "mp_min_funds": 2,
        "mp_max_funds": 9,
        "z_entry_soft": 1.5,
        "z_exit_soft": -0.75,
        "selection_metric": "Sortino",
        "weighting_name": "risk_parity",
        "lookback_periods": 4,
        "evaluation_periods": 2,
        "rebalance_freq": "Q",
    }

    expected = [
        "**Selection Parameters**",
        "- Target portfolio size: **6** funds",
        "- Min funds allowed: **2**",
        "- Max funds allowed: **9**",
        "- Selection metric: **Sortino**",
        "- Weighting: **risk_parity**",
        "**Z-Score Thresholds**",
        "- Entry threshold: **z ≥ 1.50**",
        "- Exit threshold: **z ≤ -0.75**",
        "*(Higher z-score = better relative performance)*",
        "**Time Parameters**",
        "- Rebalance frequency: **Quarterly**",
        "- In-sample (lookback): **4** periods",
        "- Out-of-sample (eval): **2** period(s)",
    ]

    page._render_selection_criteria(SimpleNamespace())

    assert stub.markdowns == expected

    stub.markdowns.clear()
    stub.session_state["model_state"]["mp_min_funds"] = 0
    page._render_selection_criteria(SimpleNamespace())

    assert stub.markdowns == expected[:2] + expected[3:]


# ---------------------------------------------------------------------------
# Manager hiring / termination timeline
# ---------------------------------------------------------------------------

GREEN = "#d4edda"
RED = "#f8d7da"
BLUE = "#cce5ff"
AMBER = "#fff3cd"


def _three_period_result() -> SimpleNamespace:
    """Alpha and Beta start; Gamma replaces Beta; Delta replaces Alpha; one candidate is skipped."""
    return SimpleNamespace(
        details={
            "period_results": [
                _period(
                    "2020-04-30",
                    "2020-06-30",
                    fund_weights={"Alpha": 0.5, "Beta": 0.5},
                    manager_changes=[
                        {"manager": "Alpha", "action": "added", "reason": "seed"},
                        {"manager": "Beta", "action": "added", "reason": "seed"},
                    ],
                ),
                _period(
                    "2020-07-31",
                    "2020-09-30",
                    fund_weights={"Alpha": 0.5, "Gamma": 0.5},
                    manager_changes=[
                        {
                            "manager": "Beta",
                            "action": "dropped",
                            "reason": "z_exit",
                            "detail": "zscore=-1.50 threshold=-1.0",
                            "firm": "FirmB",
                        },
                        {
                            "manager": "Gamma",
                            "action": "added",
                            "reason": "z_entry",
                            "detail": "zscore=1.25",
                            "firm": "FirmG",
                        },
                        {
                            "manager": "Delta",
                            "action": "skipped",
                            "reason": "min_tenure",
                            "detail": "blocked by cap",
                            "firm": "FirmD",
                        },
                    ],
                ),
                _period(
                    "2020-10-31",
                    "2020-12-31",
                    fund_weights={"Gamma": 0.5, "Delta": 0.5},
                    manager_changes=[
                        {
                            "manager": "Alpha",
                            "action": "dropped",
                            "reason": "rebalance",
                            "firm": "FirmA",
                        },
                        {
                            "manager": "Delta",
                            "action": "added",
                            "reason": "replacement",
                            "firm": "FirmD",
                        },
                    ],
                ),
            ]
        }
    )


def _weights_only_result(weights_by_period: list[dict[str, float]]) -> SimpleNamespace:
    """Consecutive quarterly periods that carry weights but no manager-change events."""
    starts = ["2020-04-30", "2020-07-31", "2020-10-31", "2021-01-31", "2021-04-30"]
    ends = ["2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31", "2021-06-30"]
    periods = [
        _period(start, end, fund_weights=weights)
        for start, end, weights in zip(starts, ends, weights_by_period)
    ]
    return SimpleNamespace(details={"period_results": periods})


def test_manager_changes_render_net_changes_decision_log_and_diagnostics(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page

    page._render_manager_changes(_three_period_result())

    assert stub.captions == [
        "Total: 6 (2 initial, 2 hired, 2 terminated, 1 skipped)",
        "Includes eligible-but-skipped candidates and policy-driven holds. "
        "Skipped entries indicate candidates that met criteria but were blocked by caps/guards.",
        "Changes by period:",
        "Unique managers: 2 initial, 2 ever hired, 2 ever terminated",
    ]
    assert stub.warning_messages == []
    assert stub.write_calls == []
    assert stub.subheaders == ["Net changes"]
    assert stub.expander_labels == [
        "Decision log (adds, drops, and skips)",
        "📊 Period-by-Period Diagnostic",
    ]
    net, decisions, by_period = stub.dataframes
    assert _frame_of(net).to_dict("records") == [
        {
            "Date": "2020-04-30",
            "Action": "Initial",
            "Manager": "Alpha",
            "Reason": "Initial portfolio selection",
        },
        {
            "Date": "2020-04-30",
            "Action": "Initial",
            "Manager": "Beta",
            "Reason": "Initial portfolio selection",
        },
        {
            "Date": "2020-07-31",
            "Action": "Hired",
            "Manager": "Gamma",
            "Reason": "Performance above threshold (z=1.25)",
        },
        {
            "Date": "2020-07-31",
            "Action": "Terminated",
            "Manager": "Beta",
            "Reason": "Performance below threshold (z=-1.50)",
        },
        {
            "Date": "2020-10-31",
            "Action": "Hired",
            "Manager": "Delta",
            "Reason": "Replaced underperformer",
        },
        {
            "Date": "2020-10-31",
            "Action": "Terminated",
            "Manager": "Alpha",
            "Reason": "Portfolio rebalance",
        },
    ]
    assert _painted_rows(net) == {0: BLUE, 1: BLUE, 2: GREEN, 3: RED, 4: GREEN, 5: RED}
    assert _frame_of(decisions).to_dict("records") == [
        {
            "Date": "2020-04-30",
            "Action": "Hired",
            "Manager": "Alpha",
            "Firm": "",
            "Reason": "Initial seed",
            "Detail": "",
        },
        {
            "Date": "2020-04-30",
            "Action": "Hired",
            "Manager": "Beta",
            "Firm": "",
            "Reason": "Initial seed",
            "Detail": "",
        },
        {
            "Date": "2020-07-31",
            "Action": "Terminated",
            "Manager": "Beta",
            "Firm": "FirmB",
            "Reason": "Performance below threshold (z=-1.50)",
            "Detail": "zscore=-1.50 threshold=-1.0",
        },
        {
            "Date": "2020-07-31",
            "Action": "Hired",
            "Manager": "Gamma",
            "Firm": "FirmG",
            "Reason": "Performance above threshold (z=1.25)",
            "Detail": "zscore=1.25",
        },
        {
            "Date": "2020-07-31",
            "Action": "Skipped",
            "Manager": "Delta",
            "Firm": "FirmD",
            "Reason": "Minimum tenure guard",
            "Detail": "blocked by cap",
        },
        {
            "Date": "2020-10-31",
            "Action": "Terminated",
            "Manager": "Alpha",
            "Firm": "FirmA",
            "Reason": "Portfolio rebalance",
            "Detail": "",
        },
        {
            "Date": "2020-10-31",
            "Action": "Hired",
            "Manager": "Delta",
            "Firm": "FirmD",
            "Reason": "Replaced underperformer",
            "Detail": "",
        },
    ]
    assert _painted_rows(decisions) == {
        0: GREEN,
        1: GREEN,
        2: RED,
        3: GREEN,
        4: AMBER,
        5: RED,
        6: GREEN,
    }
    assert by_period.to_dict("index") == {
        "2020-04-30": {"Initial": 2, "Hired": 0, "Terminated": 0},
        "2020-07-31": {"Initial": 0, "Hired": 1, "Terminated": 1},
        "2020-10-31": {"Initial": 0, "Hired": 1, "Terminated": 1},
    }


def test_manager_changes_from_weights_alone_flag_a_manager_hired_twice(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    both = {"Alpha": 0.5, "Beta": 0.5}
    result = _weights_only_result([both, {"Beta": 1.0}, both, {"Beta": 1.0}, both])

    page._render_manager_changes(result)

    assert stub.captions == [
        "Total: 6 (2 initial, 2 hired, 2 terminated)",
        "No decision log entries recorded.",
        "Changes by period:",
        "Unique managers: 2 initial, 1 ever hired, 1 ever terminated",
        "Managers hired multiple times: 1",
    ]
    assert stub.write_calls == ["{'Alpha': 2}"]
    assert stub.warning_messages == []
    net, by_period = stub.dataframes
    assert _frame_of(net)["Action"].tolist() == [
        "Initial",
        "Initial",
        "Terminated",
        "Hired",
        "Terminated",
        "Hired",
    ]
    assert by_period.to_dict("index")["2021-04-30"] == {"Initial": 0, "Hired": 1, "Terminated": 0}


def test_manager_changes_list_at_most_ten_repeatedly_hired_managers(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    everyone = {f"Fund{i:02d}": 1 / 12 for i in range(11)} | {"Anchor": 1 / 12}
    result = _weights_only_result([everyone, {"Anchor": 1.0}, everyone, {"Anchor": 1.0}, everyone])

    page._render_manager_changes(result)

    assert "Managers hired multiple times: 11" in stub.captions
    assert len(stub.write_calls) == 1
    listed = ast.literal_eval(stub.write_calls[0])
    assert len(listed) == 10
    assert set(listed.values()) == {2}
    assert set(listed) < {f"Fund{i:02d}" for i in range(11)}


def test_manager_changes_total_caption_names_only_the_kinds_of_change_present(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    both = {"Alpha": 0.5, "Beta": 0.5}

    def total_for(weights_by_period: list[dict[str, float]]) -> str:
        stub.captions.clear()
        page._render_manager_changes(_weights_only_result(weights_by_period))
        return stub.captions[0]

    assert total_for([both]) == "Total: 2 (2 initial)"
    assert total_for([{"Alpha": 1.0}, both]) == "Total: 2 (1 initial, 1 hired)"
    assert total_for([both, {"Alpha": 1.0}]) == "Total: 3 (2 initial, 1 terminated)"
    assert total_for([{"Alpha": 1.0}, both, {"Beta": 1.0}]) == (
        "Total: 3 (1 initial, 1 hired, 1 terminated)"
    )
    assert stub.warning_messages == []


def test_manager_changes_warn_when_final_holdings_disagree_with_the_timeline(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    disagreement = (
        "⚠️ Data consistency check: Initial (1) + Hired (0) - Terminated (0) = 1, "
        "but final portfolio has 2 managers. This may indicate a data extraction issue."
    )

    def warnings_for(result: SimpleNamespace) -> list[str]:
        stub.warning_messages.clear()
        page._render_manager_changes(result)
        return list(stub.warning_messages)

    # The blank-named weight is counted as a final holding but never enters the timeline.
    assert warnings_for(_weights_only_result([{"Gamma": 0.5, "": 0.5}])) == [disagreement]
    assert stub.captions[0] == "Total: 1 (1 initial)"

    # Without weights the final holdings are the selected funds, counted the same two ways.
    blank_selected = _period(
        "2020-04-30", "2020-06-30", fund_weights={}, selected_funds=["Gamma", " "]
    )
    assert warnings_for(SimpleNamespace(details={"period_results": [blank_selected]})) == [
        disagreement
    ]
    two_selected = _period(
        "2020-04-30", "2020-06-30", fund_weights={}, selected_funds=["Gamma", "Delta"]
    )
    assert warnings_for(SimpleNamespace(details={"period_results": [two_selected]})) == []

    # Equal weights stand in for missing realised weights, for the timeline and the final count.
    equal_weights = _period("2020-04-30", "2020-06-30", ew_weights={"Gamma": 0.5, "": 0.5})
    assert warnings_for(SimpleNamespace(details={"period_results": [equal_weights]})) == [
        disagreement
    ]

    # A portfolio emptied in the last period has no final holdings left to disagree with.
    assert warnings_for(_weights_only_result([{"Alpha": 1.0}, {}])) == []
    assert stub.captions[-4] == "Total: 2 (1 initial, 1 terminated)"


def test_manager_changes_with_only_decision_events_still_render_the_log(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    skipped = {
        "manager": "Delta",
        "action": "skipped",
        "reason": "min_tenure",
        "detail": "blocked by cap",
        "firm": "FirmD",
    }
    result = SimpleNamespace(
        details={
            "period_results": [
                _period("2020-04-30", "2020-06-30", fund_weights={}, manager_changes=[skipped])
            ]
        }
    )

    page._render_manager_changes(result)
    assert stub.captions[:3] == [
        "Total: 0 (1 skipped)",
        "No net hires or terminations recorded.",
        "Includes eligible-but-skipped candidates and policy-driven holds. "
        "Skipped entries indicate candidates that met criteria but were blocked by "
        "caps/guards.",
    ]
    assert _frame_of(stub.dataframes[0]).to_dict("records") == [
        {
            "Date": "2020-04-30",
            "Action": "Skipped",
            "Manager": "Delta",
            "Firm": "FirmD",
            "Reason": "Minimum tenure guard",
            "Detail": "blocked by cap",
        }
    ]
    assert stub.captions[-1] == "No changes to break down by period."


def test_manager_change_highlighters_paint_known_actions_and_skip_the_rest(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, _stub = results_page
    applied: list[Any] = []
    real_apply = Styler.apply

    def recording_apply(self: Styler, func: Any, *args: Any, **kwargs: Any) -> Styler:
        applied.append(func)
        return real_apply(self, func, *args, **kwargs)

    monkeypatch.setattr(Styler, "apply", recording_apply)

    page._render_manager_changes(_three_period_result())

    highlight_action, highlight_decision = applied

    def paint(highlight: Any, action: str) -> list[str]:
        return highlight(pd.Series({"Action": action, "Manager": "Alpha"}))

    assert paint(highlight_action, "Initial") == [f"background-color: {BLUE}"] * 2
    assert paint(highlight_action, "Hired") == [f"background-color: {GREEN}"] * 2
    assert paint(highlight_action, "Terminated") == [f"background-color: {RED}"] * 2
    assert paint(highlight_action, "Skipped") == ["", ""]
    assert paint(highlight_decision, "Hired") == [f"background-color: {GREEN}"] * 2
    assert paint(highlight_decision, "Terminated") == [f"background-color: {RED}"] * 2
    assert paint(highlight_decision, "Skipped") == [f"background-color: {AMBER}"] * 2
    assert paint(highlight_decision, "Initial") == ["", ""]


# ---------------------------------------------------------------------------
# Fund holding periods
# ---------------------------------------------------------------------------


def _holdings_result(*, with_returns: bool) -> SimpleNamespace:
    """Alpha is seeded and dropped in period two, Beta is hired then; Alpha trades +20% / -20%."""
    first_returns = pd.DataFrame(
        {"Alpha": [0.2, -0.2]}, index=pd.to_datetime(["2020-05-31", "2020-06-30"])
    )
    return SimpleNamespace(
        details={
            "period_results": [
                _period(
                    "2020-04-30",
                    "2020-06-30",
                    manager_changes=[{"manager": "Alpha", "action": "added", "reason": "seed"}],
                    out_sample_scaled=first_returns if with_returns else None,
                ),
                _period(
                    "2020-07-31",
                    "2020-09-30",
                    manager_changes=[
                        {"manager": "Alpha", "action": "dropped", "reason": "z_exit"},
                        {"manager": "Beta", "action": "added", "reason": "z_entry"},
                    ],
                ),
            ]
        }
    )


def test_fund_holdings_show_formatted_tenures_and_out_of_sample_risk(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page

    page._render_fund_holdings(_holdings_result(with_returns=True))

    assert stub.markdowns == ["**Holding Summary**", "**Out-of-Sample Risk Statistics**"]
    assert stub.captions == []
    summary, risk = stub.dataframes
    assert _frame_of(summary).to_dict("records") == [
        {"Manager": "Alpha", "Years Held": "2.0", "Entry": "2020-04-30", "Exit": "2020-07-31"},
        {"Manager": "Beta", "Years Held": "1.0", "Entry": "2020-07-31", "Exit": "Current"},
    ]
    assert _painted_rows(summary) == {1: BLUE}
    # Alpha's two out-of-sample months: total -4%, CAGR 0.96 ** 6 - 1, sample vol sqrt(0.08 * 12).
    assert risk.to_dict("records") == [
        {
            "Manager": "Alpha",
            "Years": "2.0",
            "CAGR": "-21.7%",
            "Vol": "98.0%",
            "Sharpe": "-0.22",
            "Sortino": "0.00",
            "IR": 0.0,
            "Max DD": "-20.0%",
        }
    ]


def test_fund_holdings_without_out_of_sample_returns_skip_the_risk_table(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page

    page._render_fund_holdings(_holdings_result(with_returns=False))

    assert stub.markdowns == ["**Holding Summary**"]
    assert len(stub.dataframes) == 1
    assert _frame_of(stub.dataframes[0])["Manager"].tolist() == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# One period in detail
# ---------------------------------------------------------------------------

SINGLE_PERIOD_TABS = [
    "📊 In-Sample Metrics",
    "✅ Selection",
    "📈 Out-of-Sample",
    "💰 Period Return",
]


def _detail(page: ModuleType, **fields: Any) -> dict[str, Any]:
    """Period data built by the page's own builder from a 2020-04-30 .. 2020-06-30 period."""
    return page._build_period_detail(_period("2020-04-30", "2020-06-30", **fields), 1)


def _complete_period_fields() -> dict[str, Any]:
    out_sample = pd.DataFrame(
        {
            "Alpha": [0.02, 0.04, 0.06],
            "Beta": [0.0, 0.02, 0.04],
            "Gamma": [0.01, 0.01, -0.02],
        },
        index=pd.to_datetime(["2020-04-30", "2020-05-31", "2020-06-30"]),
    )
    score_frame = pd.DataFrame(
        {
            "Sharpe": [0.5, 1.5, -0.25],
            "AnnualReturn": [0.05, 0.12, -0.02],
            "zscore": [1.9, 1.2, -1.3],
        },
        index=["Beta", "Alpha", "Gamma"],
    )
    return {
        "score_frame": score_frame,
        "selected_funds": ["Alpha", "Beta"],
        "fund_weights": {"Alpha": 0.6, "Beta": 0.4, "Gamma": 5e-13},
        "manager_changes": [
            {"manager": "Alpha", "action": "added", "reason": "z_entry", "detail": "zscore=1.25"},
            {"manager": "Gamma", "action": "dropped", "reason": "z_exit", "detail": "zscore=-1.30"},
        ],
        "out_sample_scaled": out_sample,
        "out_sample_stats": {
            "Alpha": {
                "vol": 0.1,
                "cagr": 0.2,
                "sharpe": 1.0,
                "sortino": 2.0,
                "information_ratio": 0.3,
                "max_drawdown": -0.05,
            }
        },
        "out_user_stats": {
            "cagr": 0.1234,
            "vol": 0.0567,
            "sharpe": 1.2345,
            "sortino": 2.0,
            "max_dd": -0.0891,
        },
        "out_ew_stats": SimpleNamespace(cagr=0.2, vol=0.1, sharpe=2.0, sortino=3.0, max_dd=-0.1),
    }


def test_single_period_renders_every_tab_for_a_complete_period(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page

    page._render_single_period(_detail(page, **_complete_period_fields()))

    assert stub.tab_groups == [SINGLE_PERIOD_TABS]
    assert stub.captions == ["In-sample window: 2019-01-31 to 2019-12-31"]
    assert stub.warning_messages == []
    assert stub.markdowns == [
        "### Period 1: 2020-04-30 to 2020-06-30",
        "**All candidates ranked by in-sample metrics** (green = selected)",
        "**Selection Decisions for This Period**",
        "**Added:**",
        "- ✅ **Alpha** — Performance above threshold (z=1.25)",
        "**Dropped:**",
        "- ❌ **Gamma** — Performance below threshold (z=-1.30)",
        "**Portfolio Weights Applied:**",
        "**Out-of-Sample Monthly Returns (Volatility-Adjusted)**",
        "**Weighted Portfolio (This Period)**",
        "**Equal-Weight Portfolio (This Period)**",
    ]
    assert stub.expander_labels == ["View monthly returns detail"]
    scores, weights, summary, monthly = stub.dataframes
    # Ranked by z-score (Beta first although Alpha has the better Sharpe), best first, with the
    # z-score column moved last and every cell formatted.
    assert _frame_of(scores).index.tolist() == ["Beta", "Alpha", "Gamma"]
    assert _frame_of(scores).to_dict("records") == [
        {"Sharpe": "0.50", "AnnualReturn": "5.0%", "zscore": "1.90"},
        {"Sharpe": "1.50", "AnnualReturn": "12.0%", "zscore": "1.20"},
        {"Sharpe": "-0.25", "AnnualReturn": "-2.0%", "zscore": "-1.30"},
    ]
    assert _painted_rows(scores) == {0: GREEN, 1: GREEN}
    assert weights.to_dict("records") == [
        {"Fund": "Alpha", "Weight": "60.0%"},
        {"Fund": "Beta", "Weight": "40.0%"},
    ]
    # Alpha: reported stats win; Beta has none, so the page computes them from its three months.
    assert summary.to_dict("records") == [
        {
            "Fund": "Alpha",
            "Total Return": "12.4%",
            "Ann. Vol": "10.0%",
            "Months": 3,
            "CAGR": "20.0%",
            "Sharpe": "1.00",
            "Sortino": "2.00",
            "Info Ratio": "0.30",
            "Max DD": "-5.0%",
        },
        {
            "Fund": "Beta",
            "Total Return": "6.1%",
            "Ann. Vol": "6.9%",
            "Months": 3,
            "CAGR": "26.6%",
            "Sharpe": "3.84",
            "Sortino": "0.00",
            "Info Ratio": "—",
            "Max DD": "0.0%",
        },
    ]
    assert monthly.to_dict("index") == {
        "2020-04": {"Alpha": "2.00%", "Beta": "0.00%"},
        "2020-05": {"Alpha": "4.00%", "Beta": "2.00%"},
        "2020-06": {"Alpha": "6.00%", "Beta": "4.00%"},
    }
    assert stub.metrics == [
        ("CAGR", "12.3%"),
        ("VOL", "5.7%"),
        ("Sharpe", "1.23"),
        ("Sortino", "2.00"),
        ("MAX DD", "-8.9%"),
        ("CAGR", "20.0%"),
        ("VOL", "10.0%"),
        ("Sharpe", "2.00"),
        ("Sortino", "3.00"),
        ("MAX DD", "-10.0%"),
    ]


def test_single_period_without_data_shows_every_placeholder(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    returns = pd.DataFrame({"Alpha": [0.01]}, index=pd.to_datetime(["2020-04-30"]))

    page._render_single_period(_detail(page))

    assert stub.captions == [
        "In-sample window: 2019-01-31 to 2019-12-31",
        "No in-sample metrics available for this period.",
        "No changes this period (portfolio unchanged).",
        "No out-of-sample returns available.",
        "No weighted portfolio stats available.",
        "No equal-weight portfolio stats available.",
    ]
    assert stub.markdowns == [
        "### Period 1: 2020-04-30 to 2020-06-30",
        "**Weighted Portfolio (This Period)**",
        "**Equal-Weight Portfolio (This Period)**",
    ]
    assert stub.dataframes == []
    assert stub.metrics == []

    # Returns without a selected fund to show them for are still "no returns".
    page._render_single_period(_detail(page, out_sample_scaled=returns, selected_funds=[]))

    assert stub.captions[6:] == [
        "In-sample window: 2019-01-31 to 2019-12-31",
        "No in-sample metrics available for this period.",
        "No changes this period (portfolio unchanged).",
        "No out-of-sample returns available.",
        "No weighted portfolio stats available.",
        "No equal-weight portfolio stats available.",
    ]
    assert stub.dataframes == []


def test_single_period_flags_months_missing_from_the_out_of_sample_window(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    returns = pd.DataFrame(
        {"Alpha": [0.01, 0.02]}, index=pd.to_datetime(["2020-04-30", "2020-07-31"])
    )
    fields = {"out_sample_scaled": returns, "selected_funds": ["Alpha"]}

    page._render_single_period(
        page._build_period_detail(_period("2020-04-30", "2020-07-31", **fields), 1)
    )

    assert stub.warning_messages == [
        "Missing months inside this out-of-sample window: 2020-05, 2020-06"
    ]

    # Without an end date there is no window to check, and the table still renders.
    stub.warning_messages.clear()
    page._render_single_period(page._build_period_detail(_period("2020-04-30", "", **fields), 1))

    assert stub.warning_messages == []
    assert stub.markdowns.count("**Out-of-Sample Monthly Returns (Volatility-Adjusted)**") == 2

    # A window the month check cannot parse is skipped silently; the rest of the tab renders.
    page._render_single_period(
        page._build_period_detail(_period("not-a-date", "also-bad", **fields), 1)
    )

    assert stub.warning_messages == []
    assert stub.markdowns.count("**Out-of-Sample Monthly Returns (Volatility-Adjusted)**") == 3
    assert len(stub.dataframes) == 6
    assert stub.dataframes[-2]["Fund"].tolist() == ["Alpha"]


def test_single_period_reads_stats_objects_and_falls_back_field_by_field(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    fields = _complete_period_fields()
    fields["out_sample_stats"] = {
        # An object (not a dict) whose zero Sharpe must be shown, not replaced by the computed one.
        "Alpha": SimpleNamespace(
            vol=0.1, cagr=0.2, sharpe=0.0, sortino=2.0, information_ratio=0.3, max_drawdown=-0.05
        ),
        # A partial mapping: only the CAGR is reported, everything else is computed.
        "Beta": {"cagr": 0.5},
    }
    fields["out_user_stats"] = {"cagr": 0.1, "maxdd": -0.05}
    fields["out_ew_stats"] = {"cagr": 0.2, "maxdd": -0.1}

    page._render_single_period(_detail(page, **fields))

    summary = stub.dataframes[2]
    assert summary.to_dict("records") == [
        {
            "Fund": "Alpha",
            "Total Return": "12.4%",
            "Ann. Vol": "10.0%",
            "Months": 3,
            "CAGR": "20.0%",
            "Sharpe": "0.00",
            "Sortino": "2.00",
            "Info Ratio": "0.30",
            "Max DD": "-5.0%",
        },
        {
            "Fund": "Beta",
            "Total Return": "6.1%",
            "Ann. Vol": "6.9%",
            "Months": 3,
            "CAGR": "50.0%",
            "Sharpe": "3.84",
            "Sortino": "0.00",
            "Info Ratio": "—",
            "Max DD": "0.0%",
        },
    ]
    # Both blocks read "maxdd" when "max_dd" is absent and skip the stats they lack.
    assert stub.metrics == [
        ("CAGR", "10.0%"),
        ("MAX DD", "-5.0%"),
        ("CAGR", "20.0%"),
        ("MAX DD", "-10.0%"),
    ]


def test_single_period_counts_only_months_with_a_return_and_blanks_the_rest(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    returns = pd.DataFrame(
        {
            "Alpha": [0.02, np.nan, 0.06],
            "Beta": [np.nan, np.nan, 0.05],
            "Gamma": [np.nan, np.nan, np.nan],
        },
        index=pd.to_datetime(["2020-04-30", "2020-05-31", "2020-06-30"]),
    )
    detail = _detail(page, out_sample_scaled=returns, selected_funds=["Alpha", "Beta", "Gamma"])

    page._render_single_period(detail)

    summary, monthly = stub.dataframes
    # Gamma has no return at all, so it has no row; Beta has one return, too few for risk stats.
    assert summary["Fund"].tolist() == ["Alpha", "Beta"]
    alpha, beta = summary.to_dict("records")
    # Alpha: only April and June have a return, 1.02 * 1.06 - 1 = 8.1% over two months.
    assert (alpha["Months"], alpha["Total Return"]) == (2, "8.1%")
    assert beta == {
        "Fund": "Beta",
        "Total Return": "5.0%",
        "Ann. Vol": "—",
        "Months": 1,
        "CAGR": "—",
        "Sharpe": "—",
        "Sortino": "—",
        "Info Ratio": "—",
        "Max DD": "—",
    }
    assert monthly.to_dict("index") == {
        "2020-04": {"Alpha": "2.00%", "Beta": "—", "Gamma": "—"},
        "2020-05": {"Alpha": "—", "Beta": "—", "Gamma": "—"},
        "2020-06": {"Alpha": "6.00%", "Beta": "5.00%", "Gamma": "—"},
    }
    assert stub.warning_messages == []


def test_single_period_computes_every_statistic_when_none_are_reported(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    fields = _complete_period_fields()
    fields["out_sample_stats"] = None
    fields["selected_funds"] = ["Alpha"]

    page._render_single_period(_detail(page, **fields))

    # 1.02 * 1.04 * 1.06 - 1 = 12.4% over three months: CAGR 1.124448 ** 4 - 1, vol 0.02 * sqrt(12).
    assert stub.dataframes[2].to_dict("records") == [
        {
            "Fund": "Alpha",
            "Total Return": "12.4%",
            "Ann. Vol": "6.9%",
            "Months": 3,
            "CAGR": "59.9%",
            "Sharpe": "8.64",
            "Sortino": "0.00",
            "Info Ratio": "—",
            "Max DD": "0.0%",
        }
    ]


def test_single_period_ranks_candidates_by_the_first_column_without_z_scores(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    score_frame = pd.DataFrame(
        {
            "Sortino": [0.5, 2.0, 1.25],
            "Sharpe": [1.0, 1.5, 0.25],
            "InformationRatio": [0.1, np.nan, -0.5],
            "AnnualReturn": [0.1, 0.2, 0.3],
            "Volatility": [0.05, 0.1, 0.15],
            "MaxDrawdown": [-0.1, -0.2, np.nan],
            "CAGR": [0.01, 0.02, 0.03],
            "Vol": [0.25, 0.5, 0.75],
            "Other": ["a", "b", "c"],
        },
        index=["F1", "F2", "F3"],
    )

    page._render_single_period(_detail(page, score_frame=score_frame, selected_funds=["F3"]))

    scores = stub.dataframes[0]
    assert _frame_of(scores).index.tolist() == ["F2", "F3", "F1"]
    assert _frame_of(scores).to_dict("records") == [
        {
            "Sortino": "2.00",
            "Sharpe": "1.50",
            "InformationRatio": "—",
            "AnnualReturn": "20.0%",
            "Volatility": "10.0%",
            "MaxDrawdown": "-20.0%",
            "CAGR": "2.0%",
            "Vol": "50.0%",
            "Other": "b",
        },
        {
            "Sortino": "1.25",
            "Sharpe": "0.25",
            "InformationRatio": "-0.50",
            "AnnualReturn": "30.0%",
            "Volatility": "15.0%",
            "MaxDrawdown": "—",
            "CAGR": "3.0%",
            "Vol": "75.0%",
            "Other": "c",
        },
        {
            "Sortino": "0.50",
            "Sharpe": "1.00",
            "InformationRatio": "0.10",
            "AnnualReturn": "10.0%",
            "Volatility": "5.0%",
            "MaxDrawdown": "-10.0%",
            "CAGR": "1.0%",
            "Vol": "25.0%",
            "Other": "a",
        },
    ]
    assert _painted_rows(scores) == {1: GREEN}


def test_single_period_selection_tab_lists_only_the_actions_that_happened(
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    skipped = {"manager": "Delta", "action": "skipped", "reason": "cooldown"}
    dropped = {"manager": "Gamma", "action": "dropped", "reason": "z_exit", "detail": "zscore=-2.0"}
    added = {"manager": "Beta", "action": "added", "reason": "cooldown"}

    page._render_single_period(_detail(page, manager_changes=[skipped, dropped]))

    assert stub.markdowns == [
        "### Period 1: 2020-04-30 to 2020-06-30",
        "**Selection Decisions for This Period**",
        "**Dropped:**",
        "- ❌ **Gamma** — Performance below threshold (z=-2.00)",
        "**Weighted Portfolio (This Period)**",
        "**Equal-Weight Portfolio (This Period)**",
    ]

    stub.markdowns.clear()
    page._render_single_period(_detail(page, manager_changes=[skipped, added]))

    assert stub.markdowns == [
        "### Period 1: 2020-04-30 to 2020-06-30",
        "**Selection Decisions for This Period**",
        "**Added:**",
        "- ✅ **Beta** — Cooldown guard",
        "**Weighted Portfolio (This Period)**",
        "**Equal-Weight Portfolio (This Period)**",
    ]


@pytest.mark.parametrize("stats_field", ["out_user_stats", "out_ew_stats"])
def test_single_period_return_tab_shows_the_max_drawdown_of_engine_stats(
    results_page: tuple[ModuleType, PageStub],
    stats_field: str,
) -> None:
    from trend_analysis.stages.portfolio import _Stats

    page, stub = results_page
    engine_stats = _Stats(
        cagr=0.1, vol=0.05, sharpe=1.5, sortino=2.0, max_drawdown=-0.08, information_ratio=0.3
    )

    page._render_single_period(_detail(page, **{stats_field: engine_stats}))

    assert stub.metrics == [
        ("CAGR", "10.0%"),
        ("VOL", "5.0%"),
        ("Sharpe", "1.50"),
        ("Sortino", "2.00"),
        ("MAX DD", "-8.0%"),
    ]


@pytest.mark.parametrize("stats_field", ["out_user_stats", "out_ew_stats"])
@pytest.mark.parametrize("key", ["max_drawdown", "max_dd", "maxdd"])
@pytest.mark.parametrize("drawdown, expected", [(-0.08, "-8.0%"), (0.0, "0.0%")])
def test_single_period_drawdown_dict_aliases_preserve_zero(
    results_page: tuple[ModuleType, PageStub],
    stats_field: str,
    key: str,
    drawdown: float,
    expected: str,
) -> None:
    page, stub = results_page
    page._render_single_period(_detail(page, **{stats_field: {key: drawdown}}))
    assert stub.metrics == [("MAX DD", expected)]


@pytest.mark.parametrize("stats_field", ["out_user_stats", "out_ew_stats"])
@pytest.mark.parametrize("canonical, expected", [(None, "-8.0%"), (0.0, "0.0%")])
def test_single_period_drawdown_prefers_canonical_unless_none(
    results_page: tuple[ModuleType, PageStub],
    stats_field: str,
    canonical: float | None,
    expected: str,
) -> None:
    page, stub = results_page
    stats = {"max_drawdown": canonical, "max_dd": -0.08, "maxdd": -0.12}
    page._render_single_period(_detail(page, **{stats_field: stats}))
    assert stub.metrics == [("MAX DD", expected)]


# ---------------------------------------------------------------------------
# Period-by-period breakdown
# ---------------------------------------------------------------------------


def _two_period_result() -> SimpleNamespace:
    return SimpleNamespace(
        details={
            "period_results": [
                _period("2020-04-30", "2020-06-30", fund_weights={"Alpha": 0.5, "Beta": 0.5}),
                _period("2020-07-31", "2020-09-30", fund_weights={"Alpha": 0.25, "Gamma": 0.75}),
            ]
        }
    )


def test_period_breakdown_without_periods_says_so(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    monkeypatch.setattr(
        page, "_render_single_period", lambda _data: pytest.fail("no period to render")
    )

    page._render_period_breakdown(SimpleNamespace(details={}))
    page._render_period_breakdown(SimpleNamespace(details={"period_results": []}))

    assert stub.captions == ["No period-by-period data available."] * 2
    assert stub.selectbox_calls == []
    assert stub.expander_labels == []


def test_period_breakdown_lists_periods_and_renders_the_selected_one(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    rendered: list[dict[str, Any]] = []
    monkeypatch.setattr(page, "_render_single_period", rendered.append)

    page._render_period_breakdown(_two_period_result())

    assert stub.expander_labels == ["View portfolio weights by rebalance date"]
    assert stub.dataframes[0].to_dict("index") == {
        "2020-04-30": {"Alpha": "50.0%", "Beta": "50.0%", "Gamma": "0.0%"},
        "2020-07-31": {"Alpha": "25.0%", "Beta": "0.0%", "Gamma": "75.0%"},
    }
    assert stub.selectbox_calls == [("Select period to examine:", [0, 1])]
    assert stub.selectbox_format_labels == [
        ["Period 1: 2020-04-30 to 2020-06-30", "Period 2: 2020-07-31 to 2020-09-30"]
    ]
    assert [(data["period_num"], data["out_start"], data["out_end"]) for data in rendered] == [
        (1, "2020-04-30", "2020-06-30")
    ]

    stub.selectbox_map["Select period to examine:"] = 1
    page._render_period_breakdown(_two_period_result())

    assert [data["period_num"] for data in rendered] == [1, 2]
    assert rendered[1]["fund_weights"] == {"Alpha": 0.25, "Gamma": 0.75}

    # A cleared selection renders no period at all.
    stub.selectbox_map["Select period to examine:"] = None
    page._render_period_breakdown(_two_period_result())

    assert [data["period_num"] for data in rendered] == [1, 2]


def test_period_breakdown_survives_a_failing_weights_table(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    rendered: list[dict[str, Any]] = []
    monkeypatch.setattr(page, "_render_single_period", rendered.append)

    def explode(_result: Any) -> pd.DataFrame:
        raise RuntimeError("weights table unavailable")

    monkeypatch.setattr(page, "_build_period_weights_df", explode)

    page._render_period_breakdown(_two_period_result())

    assert stub.expander_labels == []
    assert stub.dataframes == []
    assert stub.selectbox_calls == [("Select period to examine:", [0, 1])]
    assert [data["period_num"] for data in rendered] == [1]


def test_period_breakdown_hides_the_weights_table_when_no_weights_exist(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    rendered: list[dict[str, Any]] = []
    monkeypatch.setattr(page, "_render_single_period", rendered.append)
    result = SimpleNamespace(details={"period_results": [_period("2020-04-30", "2020-06-30")]})

    page._render_period_breakdown(result)

    assert stub.expander_labels == []
    assert stub.selectbox_format_labels == [["Period 1: 2020-04-30 to 2020-06-30"]]
    assert [data["period_num"] for data in rendered] == [1]


# ---------------------------------------------------------------------------
# Download section
# ---------------------------------------------------------------------------

WORKBOOK_LABEL = "⬇️ Download Phase-1 Workbook (XLSX)"
CSV_BUNDLE_LABEL = "⬇️ Download Phase-1 CSV Bundle (ZIP)"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
WORKBOOK_CAPTION = (
    "Downloads a Phase-1 style workbook including summary + per-period sheets, "
    "with out-of-sample return/risk metrics and the requested number formats."
)
CSV_BUNDLE_CAPTION = (
    "Downloads CSV files with the same content as the Excel workbook: "
    "summary metrics, per-period data, execution metrics, and manager changes."
)


def _install_export_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, produce: bool
) -> dict[str, Any]:
    """Replace the Excel/CSV-bundle exporters; the temp workbook is confined to ``tmp_path``."""
    import trend_analysis.export as export_module

    seen: dict[str, Any] = {"paths": [], "periods": []}
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def fake_workbook(period_results: list[dict[str, Any]], path: str) -> None:
        seen["paths"].append(Path(path))
        seen["periods"].append(period_results)
        Path(path).write_bytes(b"workbook-for-%d-periods" % len(period_results) if produce else b"")

    def fake_frames(period_results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
        if not produce:
            return {}
        return {
            "summary": pd.DataFrame({"OS CAGR": [12.5], "OS Sharpe": [1.5], "Fund": ["Alpha"]}),
            "empty": pd.DataFrame(),
        }

    monkeypatch.setattr(export_module, "export_phase1_workbook", fake_workbook)
    monkeypatch.setattr(export_module, "flat_frames_from_results", fake_frames)
    return seen


def _export_result() -> SimpleNamespace:
    """Alpha (period one) is replaced by Beta (period two); both trade +20% then -20%."""
    index_one = pd.to_datetime(["2020-05-31", "2020-06-30"])
    index_two = pd.to_datetime(["2020-08-31", "2020-09-30"])
    return SimpleNamespace(
        details={
            "period_results": [
                _period(
                    "2020-04-30",
                    "2020-06-30",
                    fund_weights={"Alpha": 1.0},
                    manager_changes=[{"manager": "Alpha", "action": "added", "reason": "seed"}],
                    out_sample_scaled=pd.DataFrame({"Alpha": [0.2, -0.2]}, index=index_one),
                ),
                _period(
                    "2020-07-31",
                    "2020-09-30",
                    fund_weights={"Beta": 1.0},
                    manager_changes=[
                        {"manager": "Alpha", "action": "dropped", "reason": "z_exit"},
                        {"manager": "Beta", "action": "added", "reason": "z_entry"},
                    ],
                    out_sample_scaled=pd.DataFrame({"Beta": [0.2, -0.2]}, index=index_two),
                ),
            ]
        }
    )


def test_download_section_offers_every_csv_workbook_and_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    seen = _install_export_fakes(monkeypatch, tmp_path, produce=True)
    result = _export_result()

    page._render_download_section(result)

    assert [(d["label"], d["file_name"], d["mime"]) for d in stub.downloads] == [
        ("📊 Weights", "portfolio_weights.csv", "text/csv"),
        ("📈 Returns", "portfolio_returns.csv", "text/csv"),
        ("🔄 Changes", "manager_changes.csv", "text/csv"),
        ("📋 Holdings", "fund_holdings.csv", "text/csv"),
        (WORKBOOK_LABEL, "trend_phase1_workbook.xlsx", XLSX_MIME),
        (CSV_BUNDLE_LABEL, "trend_phase1_csv_bundle.zip", "application/zip"),
    ]
    weights, returns, changes, holdings, workbook, bundle = stub.downloads
    assert weights["data"] == (
        "Period Start,Period End,Fund,Weight\n"
        "2020-04-30,2020-06-30,Alpha,100.0%\n"
        "2020-07-31,2020-09-30,Beta,100.0%\n"
    )
    assert returns["data"] == (
        "index,Alpha,Period Start,Period End,Beta\n"
        "2020-05-31,20.00%,2020-04-30,2020-06-30,\n"
        "2020-06-30,-20.00%,2020-04-30,2020-06-30,\n"
        "2020-08-31,,2020-07-31,2020-09-30,20.00%\n"
        "2020-09-30,,2020-07-31,2020-09-30,-20.00%\n"
    )
    assert changes["data"] == (
        "Date,Action,Manager,Reason\n"
        "2020-04-30,Initial,Alpha,Initial portfolio selection\n"
        "2020-07-31,Hired,Beta,Performance above threshold\n"
        "2020-07-31,Terminated,Alpha,Performance below threshold\n"
    )
    assert holdings["data"] == (
        "Manager,Years Held,Entry,Exit,Years,CAGR,Vol,Sharpe,Sortino,IR,Max DD\n"
        "Alpha,2.0,2020-04-30,2020-07-31,2.0,-21.7%,98.0%,-0.22,0.00,0.00,-20.0%\n"
        "Beta,1.0,2020-07-31,Current,1.0,-21.7%,98.0%,-0.22,0.00,0.00,-20.0%\n"
    )
    # The workbook is built from the run's periods in a temp file that is read back and removed.
    assert seen["periods"] == [result.details["period_results"]]
    assert workbook["data"] == b"workbook-for-2-periods"
    assert seen["paths"][0].parent == tmp_path
    assert seen["paths"][0].suffix == ".xlsx"
    assert not seen["paths"][0].exists()
    archive = zipfile.ZipFile(io.BytesIO(bundle["data"]))
    assert archive.namelist() == ["trend_phase1_summary.csv"]
    assert archive.read("trend_phase1_summary.csv").decode() == (
        "OS CAGR,OS Sharpe,Fund\n12.5%,1.50,Alpha\n"
    )
    assert stub.subheaders == ["📎 Excel Workbook", "📄 CSV Export (Phase-1 Style)"]
    assert stub.captions == [
        WORKBOOK_CAPTION,
        "Auto-generated narrative summary is enabled for exports.",
        CSV_BUNDLE_CAPTION,
    ]


def test_download_section_holdings_csv_has_no_risk_columns_without_returns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    _install_export_fakes(monkeypatch, tmp_path, produce=False)

    page._render_download_section(_holdings_result(with_returns=False))

    assert [d["label"] for d in stub.downloads] == ["📋 Holdings"]
    assert stub.downloads[0]["data"] == (
        "Manager,Years Held,Entry,Exit\n"
        "Alpha,2.0,2020-04-30,2020-07-31\n"
        "Beta,1.0,2020-07-31,Current\n"
    )
    assert stub.captions == [
        "No weights",
        "No returns",
        "No changes",
        WORKBOOK_CAPTION,
        "Auto-generated narrative summary is enabled for exports.",
        "Run a multi-period analysis from the Model page to enable workbook export.",
        CSV_BUNDLE_CAPTION,
        "Run a multi-period analysis from the Model page to enable CSV export.",
    ]


def test_download_section_without_a_run_offers_nothing_and_follows_the_narrative_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results_page: tuple[ModuleType, PageStub],
) -> None:
    page, stub = results_page
    seen = _install_export_fakes(monkeypatch, tmp_path, produce=True)

    page._render_download_section(SimpleNamespace(details={}), include_narrative=False)

    assert stub.downloads == []
    assert seen["paths"] == []
    assert stub.captions == [
        "No weights",
        "No returns",
        "No changes",
        "No holdings",
        WORKBOOK_CAPTION,
        "Auto-generated narrative summary is disabled for exports.",
        "Run a multi-period analysis from the Model page to enable workbook export.",
        CSV_BUNDLE_CAPTION,
        "Run a multi-period analysis from the Model page to enable CSV export.",
    ]

    stub.captions.clear()
    page._render_download_section(SimpleNamespace(details={}), include_narrative=True)

    assert stub.captions[5] == "Auto-generated narrative summary is enabled for exports."


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _record_charts(
    monkeypatch: pytest.MonkeyPatch, page: ModuleType
) -> list[tuple[str, pd.Series]]:
    """Replace the five chart builders with recorders that return a ``<name>`` tag."""
    calls: list[tuple[str, pd.Series]] = []
    for name in (
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ):

        def record(series: pd.Series, name: str = name) -> str:
            calls.append((name, series))
            return f"<{name}>"

        monkeypatch.setattr(page.charts, name, record)
    return calls


def test_charts_take_returns_turnover_and_exposures_from_the_analysis_object(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_charts(monkeypatch, page)
    returns = pd.Series([0.1, -0.1, 0.1], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    turnover = pd.Series([0.25], index=pd.to_datetime(["2020-02-29"]))
    exposures = pd.Series({"Alpha": 0.6, "Beta": 0.4})
    analysis = SimpleNamespace(returns=returns, turnover=turnover, exposures=exposures)

    page._render_charts(SimpleNamespace(analysis=analysis, details={}))

    assert stub.subheaders == [
        "Cumulative Portfolio Returns",
        "Drawdown",
        "Rolling Sharpe (12m)",
        "Turnover",
        "Current Exposures",
    ]
    assert stub.altair_payloads == [
        "<equity_chart>",
        "<drawdown_chart>",
        "<rolling_sharpe_chart>",
        "<turnover_chart>",
        "<exposure_chart>",
    ]
    by_name = dict(calls)
    # Growth of 1: 1.1, 1.1 * 0.9 = 0.99, 0.99 * 1.1 = 1.089; drawdown measured from the 1.1 peak.
    assert by_name["equity_chart"].tolist() == pytest.approx([1.1, 0.99, 1.089])
    assert by_name["drawdown_chart"].tolist() == pytest.approx([0.0, -0.1, -0.01])
    assert by_name["rolling_sharpe_chart"].empty
    assert by_name["turnover_chart"] is turnover
    assert by_name["exposure_chart"] is exposures
    assert stub.captions == []
    assert stub.info_messages == []


def test_charts_without_any_returns_ask_for_an_analysis_run(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_charts(monkeypatch, page)

    page._render_charts(SimpleNamespace(details={}))
    page._render_charts(SimpleNamespace(portfolio=pd.Series(dtype=float), details={}))

    assert stub.info_messages == ["Run the analysis to see portfolio charts."] * 2
    assert stub.subheaders == []
    assert stub.altair_payloads == []
    assert calls == []


# ---------------------------------------------------------------------------
# The page's main flow
# ---------------------------------------------------------------------------


def _funds_frame(*columns: str) -> pd.DataFrame:
    """Three months of returns for the given columns (the values never matter to these tests)."""
    index = pd.date_range("2023-01-31", periods=3, freq="ME")
    return pd.DataFrame({name: [0.01, -0.005, 0.012] for name in columns}, index=index)


def _minimal_result(**fields: Any) -> SimpleNamespace:
    """A finished run with no period detail, no analysis object and no portfolio series."""
    values: dict[str, Any] = {
        "metrics": pd.DataFrame({"Sharpe": [1.23]}),
        "details": {},
        "fallback_info": None,
    }
    values.update(fields)
    return SimpleNamespace(**values)


def _seed_page(
    page: ModuleType,
    stub: PageStub,
    *,
    result: Any = None,
    returns_df: pd.DataFrame | None = None,
    model_state: dict[str, Any] | None = None,
    **session: Any,
) -> str:
    """Fill the session so ``render_results_page`` runs; a given ``result`` is served from cache."""
    frame = _funds_frame("FundA", "FundB") if returns_df is None else returns_df
    state = {"trend_spec": {"window": 63, "lag": 1}} if model_state is None else model_state
    stub.session_state.update(
        {
            "model_state": state,
            "selected_benchmark": None,
            "data_fingerprint": "fp",
            "returns_df": frame,
            "schema_meta": {},
            "upload_status": "success",
            "analysis_fund_columns": list(frame.columns),
        }
    )
    stub.session_state.update(session)
    run_key = page._current_run_key(state, stub.session_state["selected_benchmark"])
    if result is not None:
        stub.session_state["analysis_result"] = result
        stub.session_state["analysis_result_key"] = run_key
    return run_key


def _forbid_analysis_runs(monkeypatch: pytest.MonkeyPatch, page: ModuleType) -> None:
    monkeypatch.setattr(
        page.analysis_runner,
        "run_analysis",
        lambda *_args, **_kwargs: pytest.fail("a cached result must be rendered without a re-run"),
    )


def _record_renderers(monkeypatch: pytest.MonkeyPatch, page: ModuleType) -> dict[str, list[Any]]:
    """Replace the heavy tab renderers (tested on their own) with recorders of their arguments."""
    calls: dict[str, list[Any]] = {
        "period_breakdown": [],
        "manager_changes": [],
        "fund_holdings": [],
        "download": [],
        "charts": [],
        "trailing": [],
    }
    monkeypatch.setattr(page, "_render_period_breakdown", calls["period_breakdown"].append)
    monkeypatch.setattr(page, "_render_manager_changes", calls["manager_changes"].append)
    monkeypatch.setattr(page, "_render_fund_holdings", calls["fund_holdings"].append)
    monkeypatch.setattr(page, "_render_charts", calls["charts"].append)
    monkeypatch.setattr(page, "_render_trailing_stats", calls["trailing"].append)
    monkeypatch.setattr(
        page,
        "_render_download_section",
        lambda result, *, include_narrative=True: calls["download"].append(
            (result, include_narrative)
        ),
    )
    return calls


def test_results_page_asks_for_data_then_for_a_configured_model(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    _forbid_analysis_runs(monkeypatch, page)

    page.render_results_page()

    assert stub.title_calls == ["Results"]
    assert stub.error_messages == ["Load data on the Data page before viewing results."]

    stub.session_state["returns_df"] = _funds_frame("FundA")
    stub.error_messages.clear()
    page.render_results_page()

    assert stub.error_messages == ["Configure the model before generating results."]
    assert stub.captions == []
    assert stub.button_labels == []

    stub.session_state["model_state"] = "not-a-mapping"
    stub.error_messages.clear()
    page.render_results_page()

    assert stub.error_messages == ["Configure the model before generating results."]


def test_results_page_runs_on_the_selected_funds_and_warns_about_removed_columns(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    _record_renderers(monkeypatch, page)
    runs: list[dict[str, Any]] = []

    def fake_run(df: pd.DataFrame, model_state: dict[str, Any], benchmark: Any, **kwargs: Any):
        runs.append(
            {"columns": list(df.columns), "state": dict(model_state), "benchmark": benchmark}
        )
        return _minimal_result()

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)
    frame = _funds_frame("Alpha", "Beta", "SPX", "RF")
    _seed_page(
        page,
        stub,
        returns_df=frame,
        analysis_fund_columns=["Alpha", "Beta", "SPX", "RF"],
        selected_benchmark="SPX",
        selected_risk_free="RF",
    )

    page.render_results_page()

    assert runs == [
        {
            "columns": ["Alpha", "Beta", "RF", "SPX"],
            "state": {"trend_spec": {"window": 63, "lag": 1}, "risk_free_column": "RF"},
            "benchmark": "SPX",
        }
    ]
    assert stub.captions[0] == "Using 2 selected funds for analysis"
    assert stub.warning_messages == ["Removed non-investable columns from fund selection: RF, SPX"]

    # Opposite state: a selection of plain funds is not flagged.
    stub.warning_messages.clear()
    stub.captions.clear()
    stub.session_state["analysis_fund_columns"] = ["Alpha", "Beta"]
    page.render_results_page()

    assert stub.captions[0] == "Using 2 selected funds for analysis"
    assert stub.warning_messages == []


def test_results_page_banner_reports_the_universe_membership_only_when_tracked(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    _seed_page(page, stub, result=_minimal_result())
    key = page.UNIVERSE_MEMBERSHIP_SUMMARY_KEY

    def membership_captions() -> list[str]:
        stub.captions.clear()
        page.render_results_page()
        return [text for text in stub.captions if text.startswith("Universe membership")]

    assert membership_captions() == []
    assert stub.success_messages == ["Results loaded — 2 funds."]

    stub.session_state[key] = {"fund_count": 5, "exited_fund_count": 2}
    assert membership_captions() == [
        "Universe membership applied: 5 fund(s) tracked; 2 with exit windows."
    ]

    stub.session_state[key] = {"fund_count": 3}
    assert membership_captions() == [
        "Universe membership applied: 3 fund(s) tracked; 0 with exit windows."
    ]

    stub.session_state[key] = {"fund_count": 0, "exited_fund_count": 4}
    assert membership_captions() == []

    stub.session_state[key] = ["fund_count"]
    assert membership_captions() == []


def test_results_page_run_without_a_result_asks_to_run_again(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    monkeypatch.setattr(page.analysis_runner, "run_analysis", lambda *_args, **_kwargs: None)
    _seed_page(page, stub)

    page.render_results_page()

    assert stub.markdowns == ["Run the analysis to generate performance and risk diagnostics."]
    assert stub.info_messages == ["Click Run Analysis to generate a report."]
    assert stub.success_messages == []
    assert stub.error_messages == []
    assert stub.tab_groups == []
    assert stub.session_state["analysis_result"] is None
    assert calls["trailing"] == []


@pytest.mark.parametrize(
    ("fallback_info", "expected"),
    [
        (
            {"engine": "risk_parity"},
            ["Weight engine fallback: using equal weights because risk_parity failed."],
        ),
        ({}, ["Weight engine fallback: using equal weights because unknown engine failed."]),
        (None, []),
        ("risk_parity failed", []),
    ],
    ids=["named-engine", "unnamed-engine", "no-fallback", "not-a-mapping"],
)
def test_results_page_warns_when_the_weight_engine_fell_back(
    monkeypatch: pytest.MonkeyPatch,
    results_page: tuple[ModuleType, PageStub],
    fallback_info: Any,
    expected: list[str],
) -> None:
    page, stub = results_page
    _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    _seed_page(page, stub, result=_minimal_result(fallback_info=fallback_info))

    page.render_results_page()

    assert stub.warning_messages == expected
    assert stub.error_messages == []
    assert len(stub.tab_groups) == 1


def test_results_page_summary_prefers_the_analysis_returns_and_shows_headline_metrics(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    returns = _year_of_returns()
    decoy = pd.Series([0.5] * 12, index=returns.index)
    result = _minimal_result(
        analysis=SimpleNamespace(returns=returns),
        portfolio=decoy,
        details={"portfolio_equal_weight_combined": decoy},
    )
    _seed_page(page, stub, result=result)

    page.render_results_page()

    assert len(calls["trailing"]) == 1
    assert calls["trailing"][0] is returns
    # (1.03 * 0.99) ** 6 - 1 = 12.4% over exactly one year, volatility 7.2%, Sharpe 12.4 / 7.2.
    assert stub.metrics == [
        ("Total Return", "12.4%"),
        ("CAGR", "12.4%"),
        ("Ann. Volatility", "7.2%"),
        ("Sharpe Ratio", "1.72"),
    ]
    assert stub.header_calls[0] == "Total Portfolio Performance"
    assert "Run analysis to see performance metrics." not in stub.captions

    # Two years of the same pattern: the total (1.0197 ** 12 - 1) and the annual CAGR part ways.
    stub.metrics.clear()
    two_years = pd.Series(
        [0.03, -0.01] * 12, index=pd.date_range("2019-01-31", periods=24, freq="ME")
    )
    stub.session_state["analysis_result"] = _minimal_result(
        analysis=SimpleNamespace(returns=two_years)
    )
    page.render_results_page()

    assert stub.metrics == [
        ("Total Return", "26.4%"),
        ("CAGR", "12.4%"),
        ("Ann. Volatility", "7.1%"),
        ("Sharpe Ratio", "1.75"),
    ]


def test_results_page_headline_metrics_need_twelve_months_of_returns(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    returns = _year_of_returns()
    _seed_page(page, stub, result=_minimal_result(portfolio=returns.iloc[:11]))

    page.render_results_page()

    assert len(calls["trailing"]) == 1
    assert len(calls["trailing"][0]) == 11
    assert stub.metrics == []
    assert "Run analysis to see performance metrics." not in stub.captions

    # Nothing to compute headline numbers from at all.
    calls["trailing"].clear()
    stub.session_state["analysis_result"] = _minimal_result()
    page.render_results_page()

    assert calls["trailing"] == []
    assert stub.metrics == []
    assert "Run analysis to see performance metrics." in stub.captions


def test_results_page_rebuilds_the_return_series_from_the_period_results(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    april_june = pd.date_range("2020-04-30", periods=3, freq="ME")
    july_september = pd.date_range("2020-07-31", periods=3, freq="ME")
    periods = [
        _period(
            "2020-04-30",
            "2020-06-30",
            fund_weights={"Alpha": 0.5, "Beta": 0.5},
            out_sample_scaled=pd.DataFrame(
                {"Alpha": [0.02, 0.04, 0.06], "Beta": [0.0, 0.02, 0.04]}, index=april_june
            ),
        ),
        _period(
            "2020-07-31",
            "2020-09-30",
            fund_weights={"Alpha": 1.0},
            out_sample_scaled=pd.DataFrame(
                {"Alpha": [0.1, -0.1, 0.0], "Beta": [9.0, 9.0, 9.0]}, index=july_september
            ),
        ),
    ]
    result = _minimal_result(details={"period_results": periods})
    _seed_page(page, stub, result=result)

    page.render_results_page()

    rebuilt = calls["trailing"][0]
    assert rebuilt.index.tolist() == april_june.tolist() + july_september.tolist()
    assert rebuilt.tolist() == pytest.approx([0.01, 0.03, 0.05, 0.1, -0.1, 0.0])
    assert stub.success_messages[-1] == (
        "✅ Simulation complete: **2 periods** from 2020-04-30 to 2020-09-30"
    )
    assert calls["period_breakdown"] == [result]


def test_results_page_single_period_tab_lists_the_score_frame_it_has(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    score_frame = pd.DataFrame({"Sharpe": [1.5, 0.5]}, index=["Alpha", "Beta"])
    period = _period("2020-04-30", "2020-06-30", fund_weights={"Alpha": 1.0})
    result = _minimal_result(
        period_count=1, details={"period_results": [period], "score_frame": score_frame}
    )
    _seed_page(page, stub, result=result)

    page.render_results_page()

    assert stub.info_messages == [
        "Single-period demo results include Summary and Visualizations. Run a multi-period "
        "analysis to enable Period Analysis, Fund Details, Export, and Compare.",
        "Period Analysis requires a multi-period run. Use the Model page to run more than one "
        "analysis period.",
        "Fund Details require a multi-period or custom analysis run with fund holding history.",
        "Export requires a multi-period analysis with period-level result data.",
        "Compare requires saved multi-period configurations. Save at least two Model page "
        "configurations after running a multi-period analysis.",
    ]
    assert stub.subheaders.count("In-Sample Metrics (All Candidates)") == 1
    assert [frame for frame in stub.dataframes if frame is score_frame] == [score_frame]
    assert calls["period_breakdown"] == []

    # Opposite state: no usable score frame means no table.
    stub.subheaders.clear()
    stub.dataframes.clear()
    result.details["score_frame"] = pd.DataFrame()
    page.render_results_page()

    assert "In-Sample Metrics (All Candidates)" not in stub.subheaders
    assert stub.dataframes == []


def test_results_page_without_periods_points_to_the_model_page_and_lists_the_score_frame(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    score_frame = pd.DataFrame({"Sharpe": [1.5, 0.5]}, index=["Alpha", "Beta"])
    _seed_page(page, stub, result=_minimal_result(details={"score_frame": score_frame}))

    page.render_results_page()

    assert stub.info_messages == [
        "Set Lookback/periods above 1 on the Model page for a period-by-period breakdown.",
        "Save at least two configurations on the Model page to enable A/B comparison.",
    ]
    assert stub.subheaders.count("In-Sample Metrics (All Candidates)") == 1
    assert [frame for frame in stub.dataframes if frame is score_frame] == [score_frame]
    assert calls["period_breakdown"] == []
    assert stub.tab_groups == [
        [
            "📊 Summary",
            "🔬 Period Analysis",
            "📈 Visualizations",
            "📋 Fund Details",
            "💾 Export",
            "🆚 Compare",
        ]
    ]


def test_results_page_multi_period_run_dispatches_every_tab_to_its_renderer(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    calls = _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    periods = [
        _period("2020-04-30", "2020-06-30", fund_weights={"Alpha": 1.0}),
        _period("2020-07-31", "2020-09-30", fund_weights={"Beta": 1.0}),
    ]
    result = _minimal_result(details={"period_results": periods})
    _seed_page(page, stub, result=result)

    page.render_results_page()

    assert calls["period_breakdown"] == [result]
    assert calls["manager_changes"] == [result]
    assert calls["fund_holdings"] == [result]
    assert calls["charts"] == [result]
    assert calls["download"] == [(result, True)]
    assert stub.info_messages == [
        "Save at least two configurations on the Model page to enable A/B comparison."
    ]
    assert stub.checkbox_labels == ["Generate Summary"]

    stub.session_state["export_generate_summary"] = False
    page.render_results_page()

    assert calls["download"][-1] == (result, False)


def _scored_periods() -> list[dict[str, Any]]:
    """Two periods that each rank Alpha (selected) above Beta; Beta is dropped from the weights."""

    def scores(alpha_cagr: float, gamma_cagr: float, gamma_sharpe: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "CAGR": [alpha_cagr, 0.05, gamma_cagr],
                "Sharpe": [1.5, 0.5, gamma_sharpe],
                "zscore": [1.2, 0.1, 0.3],
            },
            index=["Alpha", "Beta", "Gamma"],
        )

    return [
        _period(
            "2020-04-30",
            "2020-06-30",
            score_frame=scores(0.12, np.inf, np.nan),
            selected_funds=["Alpha"],
            fund_weights={"Alpha": 0.6, "Beta": 0.4},
        ),
        _period(
            "2020-07-31",
            "2020-09-30",
            score_frame=scores(0.2, np.nan, np.inf),
            selected_funds=["Alpha"],
            fund_weights={"Alpha": 1.0},
        ),
    ]


def test_results_page_exports_the_in_sample_metrics_of_every_period(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    result = _minimal_result(details={"period_results": _scored_periods()})
    _seed_page(page, stub, result=result)

    page.render_results_page()

    assert stub.subheaders.count("📄 Complete Period Data") == 1
    assert stub.downloads == [
        {
            "label": "📊 Download Complete Period Analysis (CSV)",
            "data": (
                "Period,In-Sample Start,In-Sample End,Out-Sample Start,Out-Sample End,Fund,"
                "Selected,Weight,InSample_CAGR,InSample_Sharpe,InSample_zscore\n"
                "1,2019-01-31,2019-12-31,2020-04-30,2020-06-30,Alpha,Yes,60.0%,12.0%,1.50,1.2\n"
                "1,2019-01-31,2019-12-31,2020-04-30,2020-06-30,Beta,No,40.0%,5.0%,0.50,0.1\n"
                "1,2019-01-31,2019-12-31,2020-04-30,2020-06-30,Gamma,No,0.0%,,,0.3\n"
                "2,2019-01-31,2019-12-31,2020-07-31,2020-09-30,Alpha,Yes,100.0%,20.0%,1.50,1.2\n"
                "2,2019-01-31,2019-12-31,2020-07-31,2020-09-30,Beta,No,0.0%,5.0%,0.50,0.1\n"
                "2,2019-01-31,2019-12-31,2020-07-31,2020-09-30,Gamma,No,0.0%,,,0.3\n"
            ),
            "file_name": "complete_period_analysis.csv",
            "mime": "text/csv",
        }
    ]
    assert (
        "Contains in-sample metrics for all candidates in all periods, "
        "with selection status and weights."
    ) in stub.captions

    # Opposite state: a multi-period run whose periods carry no score frames has nothing to export.
    stub.downloads.clear()
    stub.captions.clear()
    unscored = [
        _period("2020-04-30", "2020-06-30", fund_weights={"Alpha": 1.0}),
        _period("2020-07-31", "2020-09-30", fund_weights={"Beta": 1.0}),
    ]
    stub.session_state["analysis_result"] = _minimal_result(details={"period_results": unscored})
    page.render_results_page()

    assert stub.subheaders.count("📄 Complete Period Data") == 2
    assert stub.downloads == []
    assert not any(text.startswith("Contains in-sample metrics") for text in stub.captions)


def test_results_page_export_formats_metric_columns_by_the_words_in_their_names(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    _record_renderers(monkeypatch, page)
    _forbid_analysis_runs(monkeypatch, page)
    scores = pd.DataFrame(
        {
            "Vol": [0.1],
            "MaxDD": [0.2],
            "Max_DD": [0.3],
            "Sortino": [2.0],
            "IR": [0.5],
            "Other": [7.0],
        },
        index=["Alpha"],
    )
    periods = [
        _period(start, end, score_frame=scores, selected_funds=["Alpha"], fund_weights={"Alpha": 1})
        for start, end in [("2020-04-30", "2020-06-30"), ("2020-07-31", "2020-09-30")]
    ]
    _seed_page(page, stub, result=_minimal_result(details={"period_results": periods}))

    page.render_results_page()

    # Percentages: vol, maxdd, max_dd; ratios: sortino, ir; anything else is written as it is.
    assert [d["data"] for d in stub.downloads] == [
        "Period,In-Sample Start,In-Sample End,Out-Sample Start,Out-Sample End,Fund,Selected,Weight,"
        "InSample_Vol,InSample_MaxDD,InSample_Max_DD,InSample_Sortino,InSample_IR,InSample_Other\n"
        "1,2019-01-31,2019-12-31,2020-04-30,2020-06-30,Alpha,Yes,100.0%,10.0%,20.0%,30.0%,2.00,0.50,7.0\n"
        "2,2019-01-31,2019-12-31,2020-07-31,2020-09-30,Alpha,Yes,100.0%,10.0%,20.0%,30.0%,2.00,0.50,7.0\n"
    ]


def _arm_compare_tab(
    monkeypatch: pytest.MonkeyPatch,
    page: ModuleType,
    stub: PageStub,
    *,
    saved: dict[str, dict[str, Any]],
) -> dict[str, list[Any]]:
    """Cached run + saved configurations; the comparison runs and its renderer are recorded."""
    _record_renderers(monkeypatch, page)
    seen: dict[str, list[Any]] = {"runs": [], "rendered": []}

    def fake_run(df: pd.DataFrame, model_state: dict[str, Any], benchmark: Any, **kwargs: Any):
        seen["runs"].append((list(df.columns), dict(model_state), benchmark, kwargs))
        if model_state.get("window") == "boom":
            raise RuntimeError("provider down")
        return SimpleNamespace(tag=model_state["window"])

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)
    monkeypatch.setattr(
        page,
        "_render_comparison_results",
        lambda *args, **kwargs: seen["rendered"].append((args, kwargs)),
    )
    _seed_page(page, stub, result=_minimal_result(), saved_model_states=saved)
    return seen


def test_results_page_compare_tab_needs_two_different_saved_configurations(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    saved = {"Base": {"window": 63}, "Fast": {"window": 126}, "Slow": {"window": 252}}
    seen = _arm_compare_tab(monkeypatch, page, stub, saved=saved)

    page.render_results_page()

    names = ["Base", "Fast", "Slow"]
    assert stub.header_calls[-1] == "A/B Comparison (Saved Configurations)"
    assert stub.selectbox_calls == [("Configuration A", names), ("Configuration B", names)]
    assert stub.selectbox_keys == ["compare_config_a_tab", "compare_config_b_tab"]
    # B defaults to the second configuration, so the default pair is comparable.
    assert stub.warning_messages == []
    assert stub.button_labels == ["Re-run with custom settings", "Run A/B comparison"]

    stub.selectbox_calls.clear()
    stub.button_labels.clear()
    stub.selectbox_map.update({"Configuration A": "Slow", "Configuration B": "Slow"})
    page.render_results_page()

    assert stub.warning_messages == ["Select two different configurations to compare."]
    assert stub.button_labels == ["Re-run with custom settings"]
    assert seen["runs"] == []
    assert seen["rendered"] == []


def test_results_page_compare_runs_both_configurations_and_renders_their_overview(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    base = {"window": 63}
    fast = {"window": 126}
    seen = _arm_compare_tab(monkeypatch, page, stub, saved={"Base": base, "Fast": fast})
    stub.button_responses = [False, True]

    page.render_results_page()

    funds = ["FundA", "FundB"]
    runs = [(columns, state, benchmark) for columns, state, benchmark, _kwargs in seen["runs"]]
    assert runs == [
        (funds, {"window": 63, "risk_free_column": None}, None),
        (funds, {"window": 126, "risk_free_column": None}, None),
    ]
    data_hash = seen["runs"][0][3]["data_hash"]
    assert re.fullmatch(r"fp:[0-9a-f]{12}", data_hash)
    assert [kwargs for *_rest, kwargs in seen["runs"]] == [{"data_hash": data_hash}] * 2
    assert len(seen["rendered"]) == 1
    args, kwargs = seen["rendered"][0]
    assert args[:2] == ("Base", "Fast")
    assert (args[2].tag, args[3].tag) == (63, 126)
    assert args[4] is base
    assert args[5] is fast
    assert kwargs == {"data_hash": data_hash}
    latest = stub.session_state["compare_latest"]
    cache = stub.session_state["comparison_results_cache"]
    assert latest["configs"] == ("Base", "Fast")
    assert latest["a_key"] != latest["b_key"]
    assert cache[latest["a_key"]] == {"result": args[2], "config_name": "Base"}
    assert cache[latest["b_key"]] == {"result": args[3], "config_name": "Fast"}
    assert stub.error_messages == []


def test_results_page_compare_tab_serves_the_cached_pair_until_the_selection_changes(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    saved = {"Base": {"window": 63}, "Fast": {"window": 126}, "Slow": {"window": 252}}
    seen = _arm_compare_tab(monkeypatch, page, stub, saved=saved)
    stub.button_responses = [False, True]
    page.render_results_page()
    assert len(seen["runs"]) == 2
    assert len(seen["rendered"]) == 1
    latest = stub.session_state["compare_latest"]
    cache = stub.session_state["comparison_results_cache"]

    # No click: the pair already computed is shown again without a new run.
    stub.button_responses = [False, False]
    page.render_results_page()

    assert len(seen["runs"]) == 2
    assert len(seen["rendered"]) == 2
    assert seen["rendered"][1][0][2] is seen["rendered"][0][0][2]

    # A different second configuration has no cached pair.
    stub.button_responses = [False, False]
    stub.selectbox_map["Configuration B"] = "Slow"
    page.render_results_page()

    assert len(seen["runs"]) == 2
    assert len(seen["rendered"]) == 2

    # A missing cache entry for either side also stops the automatic render.
    stub.selectbox_map.clear()
    for dropped in ("a_key", "b_key"):
        entry = cache.pop(latest[dropped])
        stub.button_responses = [False, False]
        page.render_results_page()
        assert len(seen["rendered"]) == 2, dropped
        cache[latest[dropped]] = entry


def test_results_page_compare_failure_is_reported_and_nothing_is_rendered(
    monkeypatch: pytest.MonkeyPatch, results_page: tuple[ModuleType, PageStub]
) -> None:
    page, stub = results_page
    saved = {"Base": {"window": 63}, "Fast": {"window": "boom"}}
    seen = _arm_compare_tab(monkeypatch, page, stub, saved=saved)
    stub.button_responses = [False, True]

    page.render_results_page()

    assert [state["window"] for _cols, state, _bench, _kw in seen["runs"]] == [63, "boom"]
    assert stub.error_messages == [
        "Comparison run failed. Please verify your configurations and try again."
    ]
    assert "provider down" in stub.captions
    assert "compare_latest" not in stub.session_state
    assert seen["rendered"] == []
