from types import SimpleNamespace

import pandas as pd
import pytest

from trend.reporting import generate_unified_report
from trend_analysis.api import RunResult
from trend_analysis.signals import TrendSpec
from trend_model.spec import BacktestSpec, SampleWindow, TrendRunSpec


def _make_result() -> RunResult:
    metrics = pd.DataFrame({"Sharpe": [0.75], "CAGR": [0.12]}, index=["FundA"])
    turnover = pd.Series(
        [0.2, 0.18, 0.22],
        index=pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"]),
    )
    final_weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
    portfolio = pd.Series(
        [0.01, -0.005, 0.012],
        index=pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"]),
    )
    stats = SimpleNamespace(
        cagr=0.12,
        vol=0.09,
        sharpe=1.1,
        sortino=0.9,
        max_drawdown=-0.2,
        information_ratio=0.4,
    )
    regime_table = pd.DataFrame(
        {
            ("User", "Risk-On"): [0.12, 1.1, -0.2, 0.62, 36],
            ("User", "Risk-Off"): [0.04, 0.4, -0.1, 0.45, 18],
        },
        index=["CAGR", "Sharpe", "Max Drawdown", "Hit Rate", "Observations"],
    )
    details = {
        "out_user_stats": stats,
        "out_ew_stats": stats,
        "selected_funds": ["FundA", "FundB"],
        "risk_diagnostics": {
            "turnover": turnover,
            "final_weights": final_weights,
        },
        "performance_by_regime": regime_table,
        "regime_summary": "Risk-On windows delivered 12.0% CAGR versus 4.0% in risk-off.",
        "regime_notes": ["Synthetic regime sample"],
    }
    result = RunResult(metrics=metrics, details=details, seed=7, environment={})
    setattr(result, "portfolio", portfolio)
    return result


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        vol_adjust={"target_vol": 0.15},
        portfolio={"selection_mode": "all", "weighting_scheme": "equal"},
        run={},
        benchmarks={},
    )


def test_generate_unified_report_produces_expected_sections() -> None:
    result = _make_result()
    config = _make_config()

    artifacts = generate_unified_report(
        result, config, run_id="test123", include_pdf=False
    )

    assert "Vol-Adj Trend Analysis Report" in artifacts.html
    assert "Executive summary" in artifacts.html
    assert "Turnover" in artifacts.html
    assert "Performance by Regime" in artifacts.html
    assert "Parameter summary" in artifacts.html
    assert "Past performance does not guarantee future results" in artifacts.html
    assert artifacts.pdf_bytes is None


def test_generate_unified_report_can_emit_pdf() -> None:
    pytest.importorskip("fpdf")
    result = _make_result()
    config = _make_config()

    artifacts = generate_unified_report(result, config, run_id="abc", include_pdf=True)

    assert artifacts.pdf_bytes is not None
    assert artifacts.pdf_bytes.startswith(b"%PDF")


def test_generate_unified_report_includes_spec_summary() -> None:
    result = _make_result()
    config = _make_config()
    trend_spec = TrendSpec(window=45, lag=2, vol_adjust=True, vol_target=0.2, zscore=True)
    backtest_spec = BacktestSpec(
        window=SampleWindow("2020-01", "2020-12", "2021-01", "2021-12"),
        selection_mode="rank",
        random_n=5,
        rebalance_calendar="NYSE",
        transaction_cost_bps=10.0,
        max_turnover=None,
        rank={"inclusion_approach": "top_n", "n": 5, "score_by": "Sharpe"},
        selector={},
        weighting={},
        weighting_scheme="equal",
        custom_weights=None,
        manual_list=(),
        indices_list=(),
        benchmarks={},
        missing={},
        target_vol=0.15,
        floor_vol=None,
        warmup_periods=0,
        monthly_cost=0.0,
        previous_weights=None,
        regime={"enabled": True, "method": "rolling_return", "proxy": "SPX"},
        metrics=("Sharpe", "CAGR"),
        seed=42,
        jobs=None,
        checkpoint_dir=None,
        export_directory=None,
        export_formats=("csv",),
        output_path=None,
        output_format="csv",
        multi_period={"frequency": "ME"},
    )
    spec_bundle = TrendRunSpec(trend=trend_spec, backtest=backtest_spec, config=config)
    config._trend_run_spec = spec_bundle

    artifacts = generate_unified_report(result, config, run_id="spec", include_pdf=False)
    params = dict(artifacts.context["parameters"])

    assert "Trend window" in params
    assert "Rank inclusion" in params
