import logging

import numpy as np
import pandas as pd
import pytest

from trend_portfolio_app import sim_runner
from trend_portfolio_app.event_log import Event, EventLog
from trend_portfolio_app.policy_engine import PolicyConfig


def test_compute_score_frame_local_handles_failure(monkeypatch):
    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    )

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sim_runner.AVAILABLE_METRICS, "boom", {"fn": boom})

    df = sim_runner.compute_score_frame_local(panel.set_index("Date"))
    assert "boom" in df.columns
    assert np.isnan(df.loc["A", "boom"])


def test_compute_score_frame_local_skips_date_column():
    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        },
        index=[pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
    )

    df = sim_runner.compute_score_frame_local(panel)
    assert "Date" not in df.index


def test_compute_score_frame_validations_and_fallback(monkeypatch):
    df = pd.DataFrame({"A": [0.1, 0.2]})
    with pytest.raises(ValueError):
        sim_runner.compute_score_frame(
            df, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
        )

    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    )

    class Dummy:
        def single_period_run(self, *a, **k):
            raise ValueError("bad")

    monkeypatch.setattr(sim_runner, "HAS_TA", True)
    monkeypatch.setattr(sim_runner, "ta_pipeline", Dummy())

    sf = sim_runner.compute_score_frame(
        panel,
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    )
    assert isinstance(sf, pd.DataFrame)


def test_simresult_methods():
    ev = EventLog()
    ev.append(Event(pd.Timestamp("2020-01-31"), "hire", "A", "top_k"))
    portfolio = pd.Series(
        [0.1, -0.05], index=[pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    )
    bench = pd.Series([0.05, -0.02], index=portfolio.index)
    res = sim_runner.SimResult(
        dates=list(portfolio.index),
        portfolio=portfolio,
        weights={},
        event_log=ev,
        benchmark=bench,
    )
    assert res.portfolio_curve().iloc[-1] > 0
    assert res.drawdown_curve().min() <= 0
    assert not res.event_log_df().empty
    summary = res.summary()
    assert "information_ratio" in summary


def test_gen_review_dates_quarterly():
    df = pd.DataFrame({"A": [0.1]}, index=[pd.Timestamp("2020-01-31")])
    sim = sim_runner.Simulator(df)
    dates = sim._gen_review_dates(
        pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-30"), "q"
    )
    assert dates[0].month == 3 and dates[1].month == 6


def test_simulator_run_progress_and_fire(monkeypatch):
    data = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    ).set_index("Date")
    sim = sim_runner.Simulator(data)
    calls = []

    def cb(i, n):
        calls.append((i, n))

    first = pd.Timestamp("2020-01-31")
    second = pd.Timestamp("2020-02-29")

    def fake_compute(panel, start, end, rf_annual=0.0):
        return pd.DataFrame({"m": [1.0]}, index=["A"])

    call = {"n": 0}

    def fake_decide(
        asof, sf, current, policy, directions, cooldowns, eligible_since, tenure
    ):
        call["n"] += 1
        if call["n"] == 1:
            return {"hire": [("A", "top_k")], "fire": []}
        return {"hire": [], "fire": [("A", "bottom_k")]}

    monkeypatch.setattr(sim_runner, "compute_score_frame", fake_compute)
    monkeypatch.setattr(sim_runner, "decide_hires_fires", fake_decide)

    res = sim.run(
        start=first,
        end=second,
        freq="M",
        lookback_months=1,
        policy=PolicyConfig(top_k=1, bottom_k=1, min_track_months=0),
        progress_cb=cb,
    )
    assert calls == [(1, 2), (2, 2)]
    assert res.event_log.events[0].action == "hire"
    assert res.event_log.events[1].action == "fire"


def test_simulator_handles_equity_curve_update_failure(monkeypatch, caplog):
    first = pd.Timestamp("2020-01-31")
    second = pd.Timestamp("2020-02-29")
    data = pd.DataFrame({"A": [0.1, 0.2]}, index=[first, second])
    sim = sim_runner.Simulator(data)

    def fake_compute(panel, start, end, rf_annual=0.0):
        return pd.DataFrame({"m": [1.0]}, index=["A"])

    def fake_decide(
        asof, sf, current, policy, directions, cooldowns, eligible_since, tenure
    ):
        return {"hire": [], "fire": []}

    def fake_apply(prev_weights, target_weights, date, rb_cfg, rb_state, policy):
        # Intentionally assign invalid data to simulate equity curve update failure
        rb_state["equity_curve"] = 1
        return pd.Series(dtype=float)

    monkeypatch.setattr(sim_runner, "compute_score_frame", fake_compute)
    monkeypatch.setattr(sim_runner, "decide_hires_fires", fake_decide)
    monkeypatch.setattr(sim_runner, "_apply_rebalance_pipeline", fake_apply)

    caplog.set_level(logging.WARNING)
    sim.run(
        start=first,
        end=first,
        freq="M",
        lookback_months=1,
        policy=PolicyConfig(min_track_months=0),
    )
    assert "Failed to update equity curve" in caplog.text


def test_apply_rebalance_pipeline_no_prev():
    tw = pd.Series(dtype=float)
    res = sim_runner._apply_rebalance_pipeline(
        prev_weights=pd.Series(dtype=float),
        target_weights=tw,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg={"bayesian_only": False},
        rb_state={},
        policy=PolicyConfig(max_weight=1.0),
    )
    assert res.empty


def test_apply_rebalance_pipeline_strategies():
    prev = pd.Series({"A": 0.0, "B": 1.0, "C": 0.5})
    target = pd.Series({"A": 1.0, "B": 0.0, "C": 0.55})
    rb_state = {"since_last_reb": 0, "equity_curve": [1.0, 0.96], "guard_on": True}
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drift_band", "turnover_cap", "drawdown_guard", "unknown"],
        "params": {
            "drift_band": {"band_pct": 0.1, "mode": "full"},
            "turnover_cap": {"max_turnover": 5.0},
            "drawdown_guard": {
                "dd_threshold": 0.1,
                "guard_multiplier": 0.5,
                "recover_threshold": 0.05,
            },
        },
    }
    policy = PolicyConfig(max_weight=0.5)
    res = sim_runner._apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )
    assert isinstance(res, pd.Series)
