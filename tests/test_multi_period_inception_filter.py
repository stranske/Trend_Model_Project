from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as engine


class MPCfg:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "missing_policy": "ffill",
            "risk_free_column": "RF",
        }
        self.performance: dict[str, Any] = {"enable_cache": False}
        self.vol_adjust: dict[str, Any] = {"target_vol": 1.0}
        self.run: dict[str, Any] = {"monthly_cost": 0.0}
        self.benchmarks: dict[str, Any] = {}
        self.seed = 0

        self.portfolio: dict[str, Any] = {
            "policy": "threshold_hold",
            "rebalance_freq": "",
            "indices_list": [],
            "selector": {"params": {"rank_column": "Sharpe"}},
            "threshold_hold": {
                "metric": "Sharpe",
                "target_n": 1,
                "z_entry_soft": 1.0,
                "z_exit_soft": -1.0,
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
            },
            "constraints": {
                "max_funds": 1,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
            "weighting": {"name": "equal"},
        }

        # Multi-period config: monthly periods; allow min-history shorter than lookback.
        self.multi_period: dict[str, Any] = {
            "frequency": "M",
            "in_sample_len": 6,
            "out_sample_len": 1,
            "min_history_periods": 3,
        }

    def model_dump(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio,
            "multi_period": self.multi_period,
            "data": self.data,
        }


def _returns_with_preinception_zeros() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    # Add variation so the near-constant series guardrail does not exclude
    # otherwise-eligible funds from the universe.
    fund_a = np.array(
        [0.010, -0.005, 0.012, -0.004, 0.009, 0.011, -0.006, 0.010, 0.008, 0.007]
    )
    # FundB: pre-inception encoded as zeros through May; starts in June.
    # Post-inception values also vary so FundB isn't excluded as near-constant.
    fund_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.020, 0.010, 0.030, 0.015, 0.025])
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "FundA": fund_a,
            "FundB": fund_b,
            "RF": 0.0,
        }
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip real pipeline execution; we only care about the selection score frame.
    def fake_call_pipeline_with_diag(*args: Any, **kwargs: Any):
        del args, kwargs
        return SimpleNamespace(value={"fund_weights": {}}, diagnostic=None)

    monkeypatch.setattr(
        engine, "_call_pipeline_with_diag", fake_call_pipeline_with_diag
    )


def test_preinception_fund_excluded_until_min_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = MPCfg()

    periods = [
        # FundB inception is 2020-06-30; this period ends 2020-05-31.
        SimpleNamespace(
            in_start="2020-01",
            in_end="2020-05",
            out_start="2020-06",
            out_end="2020-07",
        ),
        # By 2020-08-31, FundB has 3 months (Jun-Aug) of history.
        SimpleNamespace(
            in_start="2020-04",
            in_end="2020-08",
            out_start="2020-09",
            out_end="2020-10",
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_returns_with_preinception_zeros())
    assert len(results) == 2

    sf1 = results[0]["selection_score_frame"]
    assert "FundA" in sf1.index
    assert "FundB" not in sf1.index

    sf2 = results[1]["selection_score_frame"]
    assert "FundA" in sf2.index
    assert "FundB" in sf2.index
