"""preprocessing.winsorise must change processed returns when enabled."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.pipeline_helpers import _attach_calendar_settings
from trend_analysis.stages.preprocessing import (
    _PreprocessStage,
    _prepare_preprocess_stage,
    _resolve_winsorise_settings,
)

OUTLIER = 5.0  # +500% monthly return


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(7)
    fund_a = rng.normal(0.01, 0.02, len(dates))
    fund_a[-1] = OUTLIER  # extreme outlier on the tip
    fund_b = rng.normal(0.005, 0.015, len(dates))
    return pd.DataFrame({"Date": dates, "FundA": fund_a, "FundB": fund_b})


def _run(enabled: bool, limits: list[float] | None = None) -> _PreprocessStage:
    cfg = {
        "data": {"frequency": "M", "timezone": "UTC"},
        "preprocessing": {
            "winsorise": {"enabled": enabled, "limits": limits or [0.05, 0.95]},
        },
    }
    df = _frame()
    _attach_calendar_settings(df, cfg)
    stage = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
    )
    assert isinstance(stage, _PreprocessStage)
    return stage


def test_winsorise_limits_clip_outlier_returns() -> None:
    off = _run(enabled=False)
    on = _run(enabled=True)

    tip_off = float(off.df["FundA"].iloc[-1])
    tip_on = float(on.df["FundA"].iloc[-1])

    assert tip_off == pytest.approx(OUTLIER)
    assert tip_on < tip_off
    expected_cap = float(off.df["FundA"].quantile(0.95))
    assert tip_on == pytest.approx(expected_cap)
    assert not on.df["FundA"].equals(off.df["FundA"])
    # Every column is clipped to its own quantile band.
    for col in ("FundA", "FundB"):
        lo, hi = off.df[col].quantile([0.05, 0.95])
        assert on.df[col].min() >= lo - 1e-12
        assert on.df[col].max() <= hi + 1e-12
    assert on.preprocess_info["winsorise"]["enabled"] is True
    assert on.preprocess_info["winsorise"]["clipped_cells"] > 0
    assert off.preprocess_info["winsorise"]["enabled"] is False
    assert off.preprocess_info["winsorise"]["clipped_cells"] == 0


def test_shipped_default_limits_still_clip_tip_outlier() -> None:
    off = _run(enabled=False)
    on = _run(enabled=True, limits=[0.01, 0.99])
    assert float(on.df["FundA"].iloc[-1]) < float(off.df["FundA"].iloc[-1])


@pytest.mark.parametrize("limits", [[0.9, 0.1], [0.5], [-0.1, 0.9], ["a", "b"]])
def test_invalid_limits_fail_loudly(limits: list) -> None:
    with pytest.raises(ValueError, match="winsorise.limits"):
        _resolve_winsorise_settings({"winsorise": {"enabled": True, "limits": limits}})


def test_disabled_ignores_limits() -> None:
    assert _resolve_winsorise_settings({"winsorise": {"enabled": False, "limits": [0.9, 0.1]}}) == {
        "enabled": False,
        "limits": None,
    }
