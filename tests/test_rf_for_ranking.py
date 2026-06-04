"""Regression tests for issue #5398 / A18.

The fund-ranking risk-free rate must honour the configured
``metrics.rf_rate_annual`` (converted to a periodic rate) even when
``metrics.rf_override_enabled`` is ``False``. Previously the ranking
``RiskStatsConfig.risk_free`` was hard-coded to ``0.0`` unless the override was
enabled, so a configured annual RF was silently ignored for ranking Sharpe and
could change which funds were selected versus user expectation.
"""

from typing import Any

import pandas as pd
import pytest

from trend_analysis import api
from trend_analysis.config import Config
from trend_analysis.diagnostics import PipelineReasonCode, pipeline_failure


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "AssetA": [0.01, 0.02],
        }
    )


def _make_config(rf_rate_annual: float, rf_override_enabled: bool) -> Config:
    return Config(
        version="1",
        data={"date_column": "Date", "frequency": "M"},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-01",
            "out_start": "2020-02",
            "out_end": "2020-02",
        },
        portfolio={},
        metrics={
            "registry": ["Sharpe"],
            "rf_rate_annual": rf_rate_annual,
            "rf_override_enabled": rf_override_enabled,
        },
        export={},
        run={},
    )


def _capture_stats_cfg(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> Any:
    captured: dict[str, Any] = {}

    def fake_single_run(*_args: Any, **kwargs: Any):
        captured.update(kwargs)
        return pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    monkeypatch.setattr(api, "_run_analysis", fake_single_run)
    api.run_simulation(cfg, _make_frame())
    assert "stats_cfg" in captured
    return captured["stats_cfg"]


def test_ranking_honors_rf_rate_annual_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override OFF + non-zero annual RF: ranking RF must be the periodic rate."""
    cfg = _make_config(rf_rate_annual=0.12, rf_override_enabled=False)
    stats_cfg = _capture_stats_cfg(monkeypatch, cfg)

    expected = (1.0 + 0.12) ** (1.0 / 12.0) - 1.0
    assert stats_cfg is not None
    # Deliberate-break gate: restoring ``rf_rate_fallback = 0.0`` for the
    # override-off path makes this assertion fail.
    assert stats_cfg.risk_free == pytest.approx(expected)
    assert stats_cfg.risk_free > 0.0


def test_ranking_rf_zero_when_annual_rate_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero configured annual RF leaves the ranking RF at exactly 0.0."""
    cfg = _make_config(rf_rate_annual=0.0, rf_override_enabled=False)
    stats_cfg = _capture_stats_cfg(monkeypatch, cfg)

    assert stats_cfg is not None
    assert stats_cfg.risk_free == pytest.approx(0.0)


def test_ranking_honors_rf_rate_annual_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override ON keeps the same periodic ranking RF (unchanged behaviour)."""
    cfg = _make_config(rf_rate_annual=0.12, rf_override_enabled=True)
    stats_cfg = _capture_stats_cfg(monkeypatch, cfg)

    expected = (1.0 + 0.12) ** (1.0 / 12.0) - 1.0
    assert stats_cfg is not None
    assert stats_cfg.risk_free == pytest.approx(expected)
