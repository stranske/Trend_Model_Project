import importlib
import logging

import numpy as np
import pandas as pd
import pytest
from trend_portfolio_app import sim_runner
from trend_portfolio_app.event_log import Event, EventLog
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import (
    SimResult,
    Simulator,
    _apply_rebalance_pipeline,
    compute_score_frame,
    compute_score_frame_local,
)


def test_import_fallback(monkeypatch):
    monkeypatch.setattr(
        sim_runner.importlib,
        "import_module",
        lambda name: exec("raise ImportError('boom')"),
    )
    importlib.reload(sim_runner)
    assert sim_runner.HAS_TA is False and sim_runner.ta_pipeline is None


def test_compute_score_frame_local_metric_error(monkeypatch):
    panel = pd.DataFrame({"A": [0.1, 0.2]}, index=pd.date_range("2020-01-31", periods=2, freq="ME"))

    def raise_bad(*args, **kwargs):
        raise ValueError("bad")

    dummy_metrics = {
        "ok": {"fn": lambda r, idx: 1.0},
        "bad": {"fn": raise_bad},
    }
    monkeypatch.setattr(sim_runner, "AVAILABLE_METRICS", dummy_metrics)
    df = compute_score_frame_local(panel)
    assert np.isnan(df.loc["A", "bad"])


def test_compute_score_frame_raises_without_date():
    df = pd.DataFrame({"A": [0.1]})
    with pytest.raises(ValueError):
        compute_score_frame(df, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"))


def test_compute_score_frame_external_errors(monkeypatch):
    df = pd.DataFrame({"Date": ["2020-01-31"], "A": [0.1]})

    class Dummy:
        pass

    dummy = Dummy()

    def bad_import(*args, **kwargs):
        raise ImportError("boom")

    dummy.single_period_run = bad_import
    monkeypatch.setattr(sim_runner, "HAS_TA", True)
    monkeypatch.setattr(sim_runner, "ta_pipeline", dummy)
    compute_score_frame(df, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"))

    def bad_value(*args, **kwargs):
        raise ValueError("bad")

    dummy.single_period_run = bad_value
    compute_score_frame(df, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"))


def test_compute_score_frame_type_error_fallback(monkeypatch, caplog):
    df = pd.DataFrame({"Date": ["2020-01-31", "2020-02-29"], "A": [0.1, 0.2]})

    class Dummy:
        pass

    def bad_type(*args, **kwargs):  # noqa: ANN001
        raise TypeError("unsupported")

    dummy = Dummy()
    dummy.single_period_run = bad_type

    monkeypatch.setattr(sim_runner, "HAS_TA", True)
    monkeypatch.setattr(sim_runner, "ta_pipeline", dummy)

    with caplog.at_level(logging.WARNING):
        result = compute_score_frame(df, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"))

    assert isinstance(result, pd.DataFrame)
    assert any("TypeError" in rec.message for rec in caplog.records)


def test_simresult_helpers():
    el = EventLog()
    el.append(Event(date=pd.Timestamp("2020-01-31"), action="hire", manager="A", reason="x"))
    sr = SimResult(
        dates=[pd.Timestamp("2020-01-31")],
        portfolio=pd.Series([0.1], index=[pd.Timestamp("2020-01-31")]),
        weights={},
        event_log=el,
        benchmark=None,
    )
    assert isinstance(sr.portfolio_curve(), pd.Series)
    assert isinstance(sr.drawdown_curve(), pd.Series)
    assert isinstance(sr.event_log_df(), pd.DataFrame)
    summary = sr.summary()
    assert "total_return" in summary


def test_simresult_bootstrap_band_cached():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.Series([0.01, -0.005, 0.012, -0.003, 0.0, 0.008], index=idx)
    sr = SimResult(
        dates=list(idx),
        portfolio=returns,
        weights={},
        event_log=EventLog(),
        benchmark=None,
    )

    sr._bootstrap_cache.clear()
    band_a = sr.bootstrap_band(n=20, block=2, random_state=123)
    assert len(sr._bootstrap_cache) == 1
    band_b = sr.bootstrap_band(n=20, block=2, random_state=123)
    assert len(sr._bootstrap_cache) == 1

    assert band_a.equals(band_b)
    assert band_a is not band_b
    assert list(band_a.columns) == ["p05", "median", "p95"]
    assert list(band_a.index) == list(sr.portfolio.index)

    band_c = sr.bootstrap_band(n=20, block=2, random_state=456)
    assert not band_c.equals(band_a)


def test_event_log_to_frame_handles_empty_and_sorting():
    """The EventLog frame output preserves shape and ordering."""

    log = EventLog()
    empty_frame = log.to_frame()
    # Empty log should still expose the schema with the date index
    assert list(empty_frame.columns) == ["action", "manager", "reason", "details"]
    assert empty_frame.index.name == "date"

    later = pd.Timestamp("2020-03-31")
    earlier = pd.Timestamp("2020-01-31")
    log.append(Event(date=later, action="hire", manager="B", reason="alpha"))
    log.append(Event(date=earlier, action="fire", manager="A", reason="beta"))

    frame = log.to_frame()
    # Events are sorted chronologically despite append order
    assert list(frame.index) == [earlier, later]
    # Original payloads preserved
    assert frame.loc[earlier, "action"] == "fire"
    assert frame.loc[later, "manager"] == "B"


def test_gen_review_dates_quarterly():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "A": [0.1, 0.2, 0.3],
        }
    )
    sim = Simulator(df)
    dates = sim._gen_review_dates(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-03-31"), "q")
    assert dates == [pd.Timestamp("2020-03-31 23:59:59.999999999")]


def test_run_progress_fire_and_equity(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "A": [0.1, -0.2],
            "B": [0.2, 0.3],
        }
    )
    sim = Simulator(df)
    calls: list[tuple[int, int]] = []

    def cb(i, total):
        calls.append((i, total))

    policy = PolicyConfig(top_k=1, bottom_k=1, min_track_months=0, metrics=[MetricSpec("m1")])

    def fake_score(panel, start, end, rf_annual=0.0):
        if end.month == 1:
            return pd.DataFrame({"m1": [1.0, -1.0]}, index=["A", "B"])
        else:
            return pd.DataFrame({"m1": [-1.0, 1.0]}, index=["A", "B"])

    monkeypatch.setattr(sim_runner, "compute_score_frame", fake_score)

    def fake_apply(prev_weights, target_weights, date, rb_cfg, rb_state, policy):
        rb_state["equity_curve"] = "bad"
        return target_weights

    monkeypatch.setattr(sim_runner, "_apply_rebalance_pipeline", fake_apply)

    result = sim.run(
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
        "M",
        1,
        policy,
        progress_cb=cb,
    )
    assert calls and calls[0] == (1, 2)
    assert any(e.action == "fire" for e in result.event_log.events)


def test_apply_rebalance_prev_empty():
    policy = PolicyConfig(max_weight=1.0)
    prev = pd.Series(dtype=float)
    tw = pd.Series({"A": 1.0})
    rb_cfg = {"bayesian_only": False}
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert list(out.index) == list(tw.index)
    assert rb_state["since_last_reb"] == 0


def test_apply_rebalance_drift_band_and_clip():
    policy = PolicyConfig(max_weight=0.4)
    prev = pd.Series({"A": 0.5, "B": 0.5})
    tw = pd.Series({"A": 0.52, "B": 0.48})
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drift_band"],
        "params": {"drift_band": {"band_pct": 0.03}},
    }
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert isinstance(out, pd.Series)


def test_apply_rebalance_drift_band_full():
    policy = PolicyConfig()
    prev = pd.Series({"A": 0.5})
    tw = pd.Series({"A": 0.0})
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drift_band"],
        "params": {"drift_band": {"band_pct": 0.01, "mode": "full", "min_trade": 0.0}},
    }
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert out.loc["A"] == 0.0


def test_apply_rebalance_turnover_cap():
    policy = PolicyConfig()
    prev = pd.Series({"A": 0.6, "B": 0.4})
    tw = pd.Series({"A": 0.5, "B": 0.5})
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["turnover_cap"],
        "params": {"turnover_cap": {"max_turnover": 1.0}},
    }
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert out.equals(tw)


def test_apply_rebalance_drawdown_guard_release():
    policy = PolicyConfig()
    prev = pd.Series({"A": 1.0})
    tw = pd.Series({"A": 1.0})
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drawdown_guard"],
        "params": {
            "drawdown_guard": {
                "dd_window": 2,
                "dd_threshold": 0.1,
                "guard_multiplier": 0.5,
                "recover_threshold": 0.05,
            }
        },
    }
    rb_state = {"equity_curve": [1.0, 1.1], "guard_on": True}
    _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert rb_state["guard_on"] is False


def test_apply_rebalance_unknown_strategy():
    policy = PolicyConfig()
    prev = pd.Series({"A": 1.0})
    tw = pd.Series({"A": 1.0})
    rb_cfg = {"bayesian_only": False, "strategies": ["unknown"]}
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert out.equals(prev)
