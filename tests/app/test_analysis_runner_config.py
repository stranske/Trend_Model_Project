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


def test_build_config_maps_min_tenure_periods(monkeypatch):
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config

    returns = pd.DataFrame(
        {"FundA": [0.01, 0.02]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    model_state = {"min_tenure_periods": 4}

    payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark=None)
    cfg = _build_config(payload)

    assert cfg.portfolio.get("min_tenure_n") == 4


def test_build_config_maps_constant_decay_to_simple(monkeypatch):
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config

    returns = pd.DataFrame(
        {"FundA": [0.01, 0.02]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    model_state = {
        "vol_adjust_enabled": True,
        "vol_window_decay": "constant",
    }

    payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark=None)
    cfg = _build_config(payload)

    assert cfg.vol_adjust["window"]["decay"] == "simple"


def test_analysis_runner_uses_canonical_config_model(monkeypatch):
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config
    from trend_analysis.config.models import Config

    returns = pd.DataFrame(
        {"FundA": [0.01, 0.02]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    payload = AnalysisPayload(returns=returns, model_state={"selection_count": 5}, benchmark=None)
    cfg = _build_config(payload)

    assert isinstance(cfg, Config)
    assert cfg.portfolio["threshold_hold"]["target_n"] == 5


def test_analysis_runner_payload_round_trips_through_canonical_config(monkeypatch):
    """Named round-trip gate for #5875: UI payload -> canonical Config sections."""

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config
    from trend_analysis.config.models import Config

    returns = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.00, 0.01],
            "FundB": [0.03, -0.01, 0.01, 0.02],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]),
    )

    model_state = {
        "selection_count": 6,
        "weighting_scheme": "risk_parity",
        "transaction_cost_bps": 7,
        "slippage_bps": 3,
        "metric_weights": {"sharpe": 0.6, "return_ann": 0.2, "drawdown": 0.2},
        "trend_window": 12,
        "trend_lag": 1,
        "trend_zscore": True,
        "trend_vol_adjust": False,
        "multi_period_enabled": True,
        "multi_period_frequency": "A",
        "lookback_periods": 1,
        "evaluation_periods": 1,
        "date_mode": "explicit",
        "start_date": "2020-02-29",
        "end_date": "2020-04-30",
        "risk_target": 0.12,
        "vol_adjust_enabled": True,
        "export_directory": "results/roundtrip",
    }

    payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark="SPX")
    cfg = _build_config(payload)

    assert isinstance(cfg, Config)
    assert cfg.portfolio["transaction_cost_bps"] == 7
    assert cfg.portfolio["weighting_scheme"] == "risk_parity"
    assert cfg.portfolio["cost_model"] == {"bps_per_trade": 7, "slippage_bps": 3}
    assert cfg.portfolio["threshold_hold"]["blended_weights"] == {
        "Sharpe": 0.6,
        "AnnualReturn": 0.2,
        "MaxDrawdown": 0.2,
    }
    assert cfg.signals["window"] == 12
    assert cfg.signals["lag"] == 1
    assert cfg.signals["zscore"] is True
    assert cfg.signals["vol_adjust"] is False
    assert cfg.sample_split == {
        "in_start": "2020-01",
        "in_end": "2020-01",
        "out_start": "2020-02",
        "out_end": "2020-04",
    }
    assert cfg.vol_adjust["target_vol"] == 0.12
    assert cfg.benchmarks["SPX"] == "SPX"
    assert cfg.multi_period == {
        "frequency": "A",
        "in_sample_len": 1,
        "out_sample_len": 1,
        "min_history_periods": 1,
        "start": "2020-02-29",
        "end": "2020-04-30",
        "start_mode": "oos",
    }
    assert cfg.export == {"directory": "results/roundtrip"}
    dumped = cfg.model_dump()
    rehydrated = Config(**dumped)
    assert rehydrated.model_dump() == dumped
    assert dumped["portfolio"]["transaction_cost_bps"] == 7
    assert dumped["portfolio"]["cost_model"]["slippage_bps"] == 3
    assert dumped["signals"]["window"] == 12
    assert dumped["sample_split"] == cfg.sample_split
    assert dumped["multi_period"]["frequency"] == "A"
    assert dumped["export"]["directory"] == "results/roundtrip"
