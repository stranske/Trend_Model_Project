import pandas as pd
import numpy as np
import pytest

import trend_portfolio_app.sim_runner as sr
from trend_portfolio_app.sim_runner import (
    Simulator,
    _apply_rebalance_pipeline,
    PolicyConfig,
)


def test_module_import_sets_has_ta_false(monkeypatch):
    import importlib
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError()))
    reloaded = importlib.reload(sr)
    assert reloaded.HAS_TA is False
    assert reloaded.ta_pipeline is None


def test_compute_score_frame_local_handles_metric_errors(monkeypatch):
    def bad_metric(r, idx):
        raise ValueError("boom")
    monkeypatch.setattr(sr, "AVAILABLE_METRICS", {"bad": {"fn": bad_metric}})
    panel = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=pd.date_range("2020-01-31", periods=3, freq="M"))
    df = sr.compute_score_frame_local(panel)
    assert np.isnan(df.loc["A", "bad"])


def test_compute_score_frame_missing_date_column():
    panel = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        sr.compute_score_frame(panel, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"))


def test_compute_score_frame_external_attribute_error(monkeypatch):
    class Fake:
        def single_period_run(self, *a, **k):
            raise AttributeError("no attr")
    monkeypatch.setattr(sr, "HAS_TA", True)
    monkeypatch.setattr(sr, "ta_pipeline", Fake())
    panel = pd.DataFrame({"Date": ["2020-01-31"], "A": [0.1]})
    out = sr.compute_score_frame(panel, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"))
    assert isinstance(out, pd.DataFrame)


def test_compute_score_frame_external_value_error(monkeypatch):
    class Fake:
        def single_period_run(self, *a, **k):
            raise ValueError("bad input")
    monkeypatch.setattr(sr, "HAS_TA", True)
    monkeypatch.setattr(sr, "ta_pipeline", Fake())
    panel = pd.DataFrame({"Date": ["2020-01-31"], "A": [0.1]})
    out = sr.compute_score_frame(panel, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"))
    assert isinstance(out, pd.DataFrame)


def test_simresult_helpers():
    ev = sr.EventLog()
    ev.append(sr.Event(date=pd.Timestamp("2020-01-31"), action="hire", manager="A", reason="top"))
    res = sr.SimResult(
        dates=[pd.Timestamp("2020-01-31")],
        portfolio=pd.Series([0.01], index=[pd.Timestamp("2020-02-29")]),
        weights={},
        event_log=ev,
        benchmark=pd.Series([0.0], index=[pd.Timestamp("2020-02-29")]),
    )
    assert isinstance(res.portfolio_curve(), pd.Series)
    assert isinstance(res.drawdown_curve(), pd.Series)
    assert isinstance(res.event_log_df(), pd.DataFrame)
    summary = res.summary()
    assert "total_return" in summary


def test_gen_review_dates_quarterly():
    df = pd.DataFrame({"A": [0.0]}, index=[pd.Timestamp("2020-01-31")])
    sim = Simulator(df)
    dates = sim._gen_review_dates(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-30"), "q")
    assert all(d.month in (3, 6) for d in dates)


def test_simulator_run_progress_and_fire(monkeypatch):
    df = pd.DataFrame({"A": [0.0, 0.0]}, index=[pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")])
    sim = Simulator(df)

    calls = []
    def progress(i, total):
        calls.append((i, total))

    decisions = [
        {"hire": [("A", "top")], "fire": []},
        {"hire": [], "fire": [("A", "bottom")]}
    ]
    def fake_decide(*a, **k):
        return decisions.pop(0)
    monkeypatch.setattr(sr, "decide_hires_fires", fake_decide)
    monkeypatch.setattr(sr, "compute_score_frame", lambda *a, **k: pd.DataFrame({"m": [1.0]}, index=["A"]))
    def stub_reb(prev_weights, target_weights, date, rb_cfg, rb_state, policy):
        rb_state["equity_curve"] = 1.0
        return target_weights
    monkeypatch.setattr(sr, "_apply_rebalance_pipeline", stub_reb)
    res = sim.run(
        start=pd.Timestamp("2020-01-31"),
        end=pd.Timestamp("2020-02-29"),
        freq="m",
        lookback_months=1,
        policy=PolicyConfig(top_k=1, bottom_k=1, min_track_months=0, metrics=[]),
        progress_cb=progress,
    )
    assert calls and calls[0][0] == 1
    assert isinstance(res, sr.SimResult)


def test_apply_rebalance_pipeline_branches(monkeypatch):
    policy = PolicyConfig(max_weight=0.5)
    # Bayesian only shortcut
    res = _apply_rebalance_pipeline(
        prev_weights=pd.Series([0.2], index=["A"]),
        target_weights=pd.Series([0.4], index=["A"]),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": True},
        rb_state={},
        policy=policy,
    )
    assert res.loc["A"] == 0.4

    # Empty previous weights branch
    res2 = _apply_rebalance_pipeline(
        prev_weights=pd.Series(dtype=float),
        target_weights=pd.Series(dtype=float),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False},
        rb_state={},
        policy=policy,
    )
    assert res2.empty

    # Drift band with clipping and continue path
    rb_state = {"since_last_reb": 0}
    res3 = _apply_rebalance_pipeline(
        prev_weights=pd.Series([0.4, 0.1], index=["A", "B"]),
        target_weights=pd.Series([0.8, 0.1], index=["A", "B"]),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False, "strategies": ["drift_band"], "params": {"drift_band": {"band_pct": 0.05, "mode": "full"}}},
        rb_state=rb_state,
        policy=policy,
    )
    assert res3.loc["A"] <= 0.5

    # Turnover cap branch where gap below threshold
    res4 = _apply_rebalance_pipeline(
        prev_weights=pd.Series([0.1], index=["A"]),
        target_weights=pd.Series([0.15], index=["A"]),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False, "strategies": ["turnover_cap"], "params": {"turnover_cap": {"max_turnover": 1.0}}},
        rb_state={"since_last_reb": 0},
        policy=policy,
    )
    assert res4.loc["A"] == pytest.approx(0.15)

    # Drawdown guard turning off
    rb_state = {"since_last_reb": 0, "equity_curve": [1.0, 1.0], "guard_on": True}
    res5 = _apply_rebalance_pipeline(
        prev_weights=pd.Series([0.2], index=["A"]),
        target_weights=pd.Series([0.2], index=["A"]),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False, "strategies": ["drawdown_guard"], "params": {"drawdown_guard": {"dd_window": 2, "dd_threshold": 0.5, "recover_threshold": 0.05}}},
        rb_state=rb_state,
        policy=policy,
    )
    assert rb_state["guard_on"] is False

    # Unknown strategy -> continue
    res6 = _apply_rebalance_pipeline(
        prev_weights=pd.Series([0.2], index=["A"]),
        target_weights=pd.Series([0.2], index=["A"]),
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False, "strategies": ["unknown"]},
        rb_state={"since_last_reb": 0},
        policy=policy,
    )
    assert res6.loc["A"] == pytest.approx(0.2)
