from __future__ import annotations

from pathlib import Path

from trend.reporting.unified import generate_unified_report
from trend_analysis import api, export
from trend_analysis.config import load

from tests.test_pipeline_integration_direct import _build_demo_config, _make_returns_frame


def test_multi_period_run_populates_performance_by_regime(tmp_path: Path) -> None:
    returns = _make_returns_frame()
    csv_path = tmp_path / "synthetic.csv"
    returns.to_csv(csv_path, index=False)

    config_data = _build_demo_config(tmp_path, csv_path)
    config_data["regime"] = {
        "enabled": True,
        "proxy": "SPX",
        "lookback": 3,
        "smoothing": 1,
        "min_obs": 2,
    }
    config = load(config_data)

    result = api.run_simulation(config, returns.copy())

    regime_table = result.details["performance_by_regime"]
    assert not regime_table.empty
    assert ("User", "All") in regime_table.columns
    assert ("Equal-Weight", "All") in regime_table.columns

    summary = export.format_summary_text(
        result.details,
        config.sample_split["in_start"],
        config.sample_split["in_end"],
        config.sample_split["out_start"],
        config.sample_split["out_end"],
    )
    assert "Performance by regime" in summary

    report = generate_unified_report(result, config, run_id="multi-period-regime")
    assert "Performance by Regime" in report.html
    assert "Regime analysis unavailable" not in report.html
