import pandas as pd

from trend.reporting import quick_summary


def test_equity_chart_reports_diagnostic_for_empty_returns():
    result = quick_summary._equity_drawdown_chart(pd.Series(dtype=float))

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == "NO_RETURNS_SERIES"


def test_turnover_chart_reports_diagnostic_for_empty_series():
    result = quick_summary._turnover_chart(pd.Series(dtype=float))

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == "NO_TURNOVER_SERIES"


def test_render_html_surfaces_equity_diagnostic():
    html = quick_summary._render_html(
        run_id="r1",
        generated_at=pd.Timestamp("2024-01-01"),
        config_text=None,
        metrics=pd.DataFrame([[1.0]], columns=["Sharpe"], index=["row"]),
        summary_text=None,
        equity_chart=None,
        turnover_chart=None,
        equity_diagnostic=quick_summary.DiagnosticPayload(
            reason_code="NO_RETURNS_SERIES",
            message="No portfolio returns available to draw equity curve.",
        ),
        turnover_diagnostic=None,
        heatmap_rel_path="heat.png",
    )

    assert "No portfolio returns available" in html
