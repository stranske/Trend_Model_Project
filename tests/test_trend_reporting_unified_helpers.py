from __future__ import annotations

from types import SimpleNamespace

import base64
import logging
from typing import Any, Mapping

import pandas as pd
import pytest

from trend.reporting import unified


def test_series_and_stats_helpers():
    series_input = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    mapping_input = {"x": 1, "y": 2}
    sequence_input = [0.5, 0.25]

    result_series = unified._coerce_series(series_input)
    assert result_series.equals(series_input)
    assert unified._coerce_series(mapping_input).sum() == pytest.approx(3.0)
    assert unified._coerce_series(sequence_input).tolist() == [0.5, 0.25]

    with pytest.raises(TypeError):
        unified._coerce_series(object())

    assert unified._maybe_series("not-convertible") is None
    assert unified._safe_float(None) is None
    assert unified._safe_float("3.14") == pytest.approx(3.14)
    assert unified._safe_float("abc") is None

    tuple_stats = unified._stats_to_dict((0.1, 0.2, 0.3, 0.4, -0.5, 0.6))
    assert tuple_stats["cagr"] == pytest.approx(0.1)

    class StatsDict:
        def __init__(self):
            self.cagr = 0.12
            self.vol = 0.09
            self.sharpe = 1.1
            self.sortino = 0.9
            self.max_drawdown = -0.2
            self.information_ratio = 0.4

    class StatsAsDict:
        def _asdict(self):
            return {"cagr": "0.5", "vol": 0.1, "sharpe": 0.8}

    dict_stats = unified._stats_to_dict(StatsDict())
    assert dict_stats["sharpe"] == pytest.approx(1.1)
    asdict_stats = unified._stats_to_dict(StatsAsDict())
    assert asdict_stats["cagr"] == pytest.approx(0.5)

    range_index = pd.RangeIndex(1)
    assert unified._periods_per_year(range_index) == 12.0
    dt_index = pd.date_range("2024-01-01", periods=4, freq="Q")
    assert unified._periods_per_year(dt_index) == 4.0
    irregular_index = pd.DatetimeIndex(
        ["2024-01-01", "2024-02-01", "2024-08-01", "2025-08-01"]
    )
    assert unified._periods_per_year(irregular_index) == 1.0
    period_index = pd.period_range("2020-01", periods=3, freq="M")
    assert unified._periods_per_year(period_index) == 12.0

    returns = pd.Series([0.0, 0.02, -0.01])
    dd_curve = unified._drawdown_curve(returns)
    assert dd_curve.min() <= 0

    assert unified._coerce_window_mode("Expanding") == "expanding"
    assert unified._coerce_window_mode("ROLL") == "rolling"
    assert unified._coerce_window_mode(123) == "rolling"

    assert unified._format_percent(0.123) == "12.3%"
    assert unified._format_percent(None) == "—"
    assert unified._format_ratio(1.2345) == "1.23"
    assert unified._format_number(1234.5) == "1,234.50"


def _build_result_with_details() -> tuple[SimpleNamespace, SimpleNamespace]:
    index = pd.period_range("2021-01", periods=6, freq="M")
    portfolio = pd.Series([0.01, -0.005, 0.012, 0.008, -0.004, 0.015], index=index)
    turnover = pd.Series(
        [0.2, 0.18, 0.22], index=pd.date_range("2021-01-31", periods=3, freq="M")
    )
    final_weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
    regime_table = pd.DataFrame(
        {"Risk-On": [0.12, 0.03, -0.2, 0.6, 24]},
        index=["CAGR", "Volatility", "Max Drawdown", "Sharpe", "Observations"],
    )
    details = {
        "portfolio_equal_weight_combined": portfolio,
        "risk_diagnostics": {"turnover": turnover, "final_weights": final_weights},
        "window_mode": "expanding",
        "window_size": 12,
        "out_user_stats": SimpleNamespace(cagr=0.1, sharpe=0.8, max_drawdown=-0.2),
        "out_ew_stats": SimpleNamespace(cagr=0.05, sharpe=0.6),
        "selected_funds": ["FundA", "FundB", "FundC"],
        "performance_by_regime": regime_table,
        "regime_summary": "Risk-on periods dominated.",
        "regime_notes": ["Synthetic insight"],
    }
    metrics = pd.DataFrame({"Sharpe": [0.7], "CAGR": [0.11]}, index=["FundA"])
    result = SimpleNamespace(details=details, portfolio=None, metrics=metrics, seed=42)
    config = SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        vol_adjust={"target_vol": 0.15, "floor_vol": 0.05, "warmup_periods": 3},
        portfolio={
            "selection_mode": "rank",
            "weighting_scheme": "equal",
            "max_turnover": 0.2,
            "rebalance_calendar": "M",
        },
        run={"monthly_cost": 5},
        benchmarks={"SPX": "S&P 500"},
        trend_spec=None,
        backtest_spec=None,
    )
    trend_spec = SimpleNamespace(
        window=63,
        lag=1,
        min_periods=20,
        vol_adjust=True,
        vol_target=0.2,
        zscore=True,
    )
    backtest_spec = SimpleNamespace(
        rank={"inclusion_approach": "top_pct", "pct": 0.2, "score_by": "Sharpe"},
        metrics=("Sharpe", "Sortino"),
        regime={"enabled": True, "method": "rolling", "proxy": "SPX"},
        multi_period={"frequency": "ME"},
    )
    spec_bundle = SimpleNamespace(trend=trend_spec, backtest=backtest_spec)
    config._trend_run_spec = spec_bundle
    return result, config


def test_build_backtest_and_context_generation():
    result, config = _build_result_with_details()
    backtest = unified._build_backtest(result)
    assert backtest is not None
    assert backtest.window_mode == "expanding"
    assert "total_return" in backtest.metrics

    exec_summary = unified._build_exec_summary(result, backtest)
    assert any("out-of-sample" in item for item in exec_summary)

    params = unified._build_param_summary(config)
    keys = dict(params)
    assert "Selection mode" in keys
    assert "Trend window" in keys

    caveats = unified._build_caveats(result, backtest)
    assert caveats == []

    metrics_html, metrics_text = unified._metrics_table_html(result.metrics)
    assert "report-table" in metrics_html
    assert any("Sharpe" in row for row in metrics_text)

    regime_html, regime_text = unified._format_regime_table(
        result.details["performance_by_regime"]
    )
    assert "Risk-On" in regime_html
    assert regime_text[0].startswith("Metric")

    narrative = unified._narrative(backtest, "Summary")
    assert "Summary" in narrative

    turnover_chart = unified._turnover_chart(backtest)
    exposure_chart = unified._exposure_chart(backtest)
    assert turnover_chart and exposure_chart

    context = {
        "title": "Test Report",
        "run_id": "run-123",
        "exec_summary": exec_summary,
        "narrative": narrative,
        "metrics_html": metrics_html,
        "metrics_text": metrics_text,
        "regime_html": regime_html,
        "regime_text": regime_text,
        "regime_summary": "Summary text",
        "regime_notes": ["Note"],
        "parameters": params,
        "caveats": ["None"],
        "turnover_chart": turnover_chart,
        "exposure_chart": exposure_chart,
        "footer": "Footer text",
    }
    html = unified._render_html(context)
    assert "Test Report" in html
    artifacts = unified.generate_unified_report(result, config, run_id="run-123")
    assert "Executive summary" in artifacts.html
    assert artifacts.pdf_bytes is None


def test_stats_to_dict_handles_problematic_inputs():
    class BadAsDict:
        def __init__(self) -> None:
            self.cagr = "0.25"
            self.vol = 0.1

        def _asdict(self) -> Mapping[str, float]:  # pragma: no cover - runtime hit
            raise RuntimeError("boom")

    mapping_stats = unified._stats_to_dict({"cagr": "0.15"})
    assert mapping_stats["cagr"] == pytest.approx(0.15)

    fallback_stats = unified._stats_to_dict(BadAsDict())
    assert fallback_stats["cagr"] == pytest.approx(0.25)

    seq_stats = unified._stats_to_dict([0.1, 0.2, 0.3, 0.4, -0.5, 0.6])
    assert seq_stats["max_drawdown"] == pytest.approx(-0.5)


def test_periods_per_year_additional_frequencies():
    weekly = pd.date_range("2024-01-07", periods=4, freq="W-SUN")
    assert unified._periods_per_year(weekly) == 52.0

    daily = pd.date_range("2024-01-01", periods=5, freq="D")
    assert unified._periods_per_year(daily) == 252.0

    uneven_fast = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"])
    assert unified._periods_per_year(uneven_fast) == 252.0

    uneven_weekly = pd.DatetimeIndex(["2024-01-01", "2024-01-05", "2024-01-13"])
    assert unified._periods_per_year(uneven_weekly) == 52.0

    uneven_monthly = pd.DatetimeIndex(["2024-01-01", "2024-01-20", "2024-02-18"])
    assert unified._periods_per_year(uneven_monthly) == 12.0

    uneven_quarterly = pd.DatetimeIndex(["2024-01-01", "2024-03-15", "2024-06-15"])
    assert unified._periods_per_year(uneven_quarterly) == 4.0

    period_daily = pd.period_range("2024-01-01", periods=3, freq="D")
    assert unified._periods_per_year(period_daily) == 252.0


def test_build_backtest_handles_fallback_portfolio():
    fallback_series = pd.Series([0.01, -0.02, 0.015], index=[0, 1, 2])
    details = {
        "portfolio": fallback_series,
        "risk_diagnostics": {"turnover": [0.05, 0.02]},
    }
    result = SimpleNamespace(details=details, portfolio=None)

    backtest = unified._build_backtest(result)
    assert backtest is not None
    assert backtest.calendar.empty
    assert backtest.turnover.empty is False


def test_build_backtest_returns_none_for_empty_series():
    details = {"portfolio": []}
    result = SimpleNamespace(details=details, portfolio=None)
    assert unified._build_backtest(result) is None


def test_rank_summary_handles_invalid_inputs():
    bad_top_n = unified._rank_summary({"inclusion_approach": "top_n", "n": object()})
    assert bad_top_n == "top_n"

    bad_pct = unified._rank_summary({"inclusion_approach": "top_pct", "pct": "oops"})
    assert bad_pct == "top_pct"

    bad_threshold = unified._rank_summary({"inclusion_approach": "threshold", "threshold": "boom"})
    assert bad_threshold == "threshold"


def test_backtest_spec_summary_handles_disabled_regime():
    spec = SimpleNamespace(
        rank={},
        metrics=("Sharpe",),
        regime={"enabled": False},
        multi_period={"frequency": "Quarterly"},
    )
    summary = dict(unified._backtest_spec_summary(spec))
    assert summary["Regime analysis"] == "Disabled"
    assert summary["Multi-period frequency"] == "Quarterly"


def test_build_param_summary_exposes_optional_fields():
    config = SimpleNamespace(
        sample_split={"in_start": "2020-01", "out_end": "2021-12"},
        vol_adjust={"floor_vol": 0.02, "warmup_periods": 5},
        portfolio={"max_turnover": 0.1, "rebalance_calendar": "NYSE"},
        run={},
        benchmarks={"SPX": "S&P 500", "NDX": "Nasdaq"},
        trend_spec=SimpleNamespace(window=63, lag=1, vol_adjust=False, zscore=False),
        backtest_spec=SimpleNamespace(regime={"enabled": True, "method": "rolling"}, metrics=()),
    )
    params = dict(unified._build_param_summary(config))
    assert params["Out-of-sample window"].endswith("→ 2021-12")
    assert params["Floor volatility"] == "2.0%"
    assert params["Warm-up periods"] == "5"
    assert params["Turnover cap"] == "10.0%"
    assert params["Benchmarks"] == "2"


def test_build_caveats_handles_empty_selection_and_no_backtest():
    result = SimpleNamespace(
        details={"selected_funds": []},
        metrics=pd.DataFrame(),
        fallback_info={"engine": "bayes", "error": "depleted"},
    )
    caveats = unified._build_caveats(result, None)
    assert any("fallback" in item for item in caveats)
    assert any("No funds selected" in item for item in caveats)
    assert any("Backtest result unavailable" in item for item in caveats)


class _StubPDFBase:
    w = 210
    l_margin = 10
    r_margin = 10

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def set_auto_page_break(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_auto_page_break", args, kwargs))

    def add_page(self) -> None:
        self.calls.append(("add_page", (), {}))

    def set_font(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_font", args, kwargs))

    def cell(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("cell", args, kwargs))

    def ln(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("ln", args, kwargs))

    def multi_cell(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("multi_cell", args, kwargs))

    def image(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("image", args, kwargs))


def _pdf_context(turnover: str = "", exposure: str = "") -> Mapping[str, Any]:
    return {
        "title": "PDF Report",
        "run_id": "run-1",
        "exec_summary": ["Point A", "Point B"],
        "narrative": "Narrative text",
        "metrics_text": ["Name | Sharpe"],
        "regime_summary": "Summary",
        "regime_text": ["Row"],
        "regime_notes": ["Note"],
        "parameters": [("Alpha", "1")],
        "caveats": ["None"],
        "turnover_chart": turnover,
        "exposure_chart": exposure,
        "footer": "Footer text",
    }


def test_render_pdf_with_stub(monkeypatch: pytest.MonkeyPatch):
    class StubPDF(_StubPDFBase):
        def output(self, dest: str = "S") -> bytes:
            assert dest == "S"
            return b"%PDF-stub"

    fake_image = base64.b64encode(b"fake-image").decode("ascii")

    monkeypatch.setattr(unified, "_load_fpdf", lambda: StubPDF)
    pdf_bytes = unified._render_pdf(_pdf_context(fake_image, fake_image))
    assert pdf_bytes.startswith(b"%PDF")


def test_render_pdf_handles_bytearray_and_missing_turnover(monkeypatch: pytest.MonkeyPatch):
    class StubPDF(_StubPDFBase):
        def output(self, dest: str = "S") -> bytearray:
            return bytearray(b"%PDF-bytearray")

    fake_image = base64.b64encode(b"alt-image").decode("ascii")
    monkeypatch.setattr(unified, "_load_fpdf", lambda: StubPDF)
    pdf_bytes = unified._render_pdf(_pdf_context("", fake_image))
    assert pdf_bytes == b"%PDF-bytearray"


def test_render_pdf_logs_when_output_is_str(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    class StubPDF(_StubPDFBase):
        def output(self, dest: str = "S") -> str:
            return "%PDF-str"

    monkeypatch.setattr(unified, "_load_fpdf", lambda: StubPDF)
    caplog.set_level(logging.WARNING)
    pdf_bytes = unified._render_pdf(_pdf_context())
    assert pdf_bytes == b"%PDF-str"
    assert any("PDF output fallback" in record.message for record in caplog.records)


def test_generate_unified_report_with_pdf(monkeypatch: pytest.MonkeyPatch):
    class StubPDF(_StubPDFBase):
        def output(self, dest: str = "S") -> bytes:
            return b"%PDF-report"

    result, config = _build_result_with_details()
    monkeypatch.setattr(unified, "_load_fpdf", lambda: StubPDF)

    artifacts = unified.generate_unified_report(
        result, config, run_id="pdf-run", include_pdf=True
    )
    assert artifacts.pdf_bytes == b"%PDF-report"


def test_rank_summary_and_pdf_helpers():
    assert unified._rank_summary({}) == ""
    assert "n=5" in unified._rank_summary({"inclusion_approach": "top_n", "n": "5"})
    assert "pct=20%" in unified._rank_summary(
        {"inclusion_approach": "top_pct", "pct": 0.2}
    )
    assert "≥ 1.50" in unified._rank_summary(
        {"inclusion_approach": "threshold", "threshold": "1.5"}
    )

    params = [("Alpha", "1")]
    unified._extend_params(params, [("Alpha", "2"), ("Beta", "3")])
    assert dict(params)["Beta"] == "3"

    assert unified._wrap_pdf_text("", initial_indent="- ") == "-"
    wrapped = unified._wrap_pdf_text("A long sentence that should wrap", width=10)
    assert "\n" in wrapped
    assert unified._pdf_safe("text\x00with\rnoise") == "textwith noise"


def test_exec_summary_and_caveats_edge_cases():
    result = SimpleNamespace(
        details={},
        metrics=pd.DataFrame(),
        fallback_info={"engine": "hrp", "error": "boom"},
    )
    caveats = unified._build_caveats(result, None)
    assert any("fallback" in item for item in caveats)
    summary = unified._build_exec_summary(result, None)
    assert summary
    assert any(
        "Metrics table is empty" in item
        for item in unified._build_caveats(
            SimpleNamespace(details={}, metrics=pd.DataFrame()), None
        )
    )

    narrative = unified._narrative(None, "Regime note")
    assert "Regime note" in narrative
