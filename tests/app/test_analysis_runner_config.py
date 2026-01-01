from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd


def test_build_config_populates_threshold_hold_metric_and_capacity(monkeypatch):
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config

    returns = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.00],
            "FundB": [0.03, -0.01, 0.01],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )

    model_state = {
        "selection_count": 8,
        "metric_weights": {"sharpe": 0.5, "return_ann": 0.25, "drawdown": 0.25},
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "lookback_periods": 1,
        "evaluation_periods": 1,
        "date_mode": "explicit",
        "start_date": "2020-02-29",
        "end_date": "2020-03-31",
        "z_entry_soft": 1.0,
        "z_exit_soft": -0.5,
        "soft_strikes": 2,
        "entry_soft_strikes": 1,
        "min_weight_strikes": 4,
        "min_weight": 0.03,
        "mp_max_funds": 25,
        "mp_min_funds": 10,
        "cooldown_periods": 2,
        "regime_enabled": True,
        "regime_proxy": "ACWI",
    }

    payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark=None)
    cfg = _build_config(payload)

    portfolio = cfg.portfolio
    assert portfolio.get("policy") == "threshold_hold"

    th = portfolio.get("threshold_hold") or {}
    assert th.get("metric") == "blended"
    assert th.get("target_n") == 8
    assert th.get("blended_weights") == {
        "Sharpe": 0.5,
        "AnnualReturn": 0.25,
        "MaxDrawdown": 0.25,
    }

    constraints = portfolio.get("constraints") or {}
    assert constraints.get("max_funds") == 25
    assert constraints.get("min_funds") == 10
    assert constraints.get("min_weight_strikes") == 4
    assert constraints.get("min_weight") == 0.03

    assert portfolio.get("cooldown_periods") == 2

    assert cfg.regime.get("enabled") is True
    assert cfg.regime.get("proxy") == "ACWI"
