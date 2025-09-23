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


def test_compute_score_frame_local_skips_date_column(monkeypatch):
    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    )

    monkeypatch.setitem(
        sim_runner.AVAILABLE_METRICS, "dummy", {"fn": lambda r, idx: 1.0}
    )

    df = sim_runner.compute_score_frame_local(panel)
    assert "Date" not in df.index
    assert df.loc["A", "dummy"] == 1.0


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


def test_compute_score_frame_external_success(monkeypatch):
    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    )

    class Dummy:
        def single_period_run(self, panel, start, end):
            assert start == "2020-01" and end == "2020-02"
            return pd.DataFrame({"metric": [1.0]}, index=["A"])

    monkeypatch.setattr(sim_runner, "HAS_TA", True)
    monkeypatch.setattr(sim_runner, "ta_pipeline", Dummy())

    sf = sim_runner.compute_score_frame(
        panel,
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    )

    assert list(sf.index) == ["A"] and "metric" in sf.columns


def test_compute_score_frame_non_callable_external(monkeypatch):
    panel = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "A": [0.1, 0.2],
        }
    )

    class Dummy:
        single_period_run = None  # attribute present but not callable

    monkeypatch.setattr(sim_runner, "HAS_TA", True)
    monkeypatch.setattr(sim_runner, "ta_pipeline", Dummy())

    sf = sim_runner.compute_score_frame(
        panel,
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    )

    # Should fall back to local implementation when attribute isn't callable
    assert isinstance(sf, pd.DataFrame)
    assert "A" in sf.index


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
        freq="ME",
        lookback_months=1,
        policy=PolicyConfig(top_k=1, bottom_k=1, min_track_months=0),
        progress_cb=cb,
    )
    assert calls == [(1, 2), (2, 2)]
    assert res.event_log.events[0].action == "hire"
    assert res.event_log.events[1].action == "fire"


def test_simulator_run_max_weight_clip(monkeypatch):
    periods = pd.period_range("2020-01", "2020-02", freq="M")
    index = periods.to_timestamp(how="end")
    data = pd.DataFrame({"A": [0.1, 0.0], "B": [0.2, 0.0]}, index=index)
    sim = sim_runner.Simulator(data)

    monkeypatch.setattr(
        sim_runner,
        "compute_score_frame",
        lambda *a, **k: pd.DataFrame({"m": [1.0, 0.9]}, index=["A", "B"]),
    )

    call_count = {"n": 0}

    def fake_decide(
        asof, sf, current, policy, directions, cooldowns, eligible_since, tenure
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"hire": [("A", "top_k"), ("B", "top_k")], "fire": []}
        return {"hire": [], "fire": []}

    monkeypatch.setattr(sim_runner, "decide_hires_fires", fake_decide)

    captured_targets: list[pd.Series] = []

    def fake_apply(prev_weights, target_weights, date, rb_cfg, rb_state, policy):
        captured_targets.append(target_weights.copy())
        return target_weights

    monkeypatch.setattr(sim_runner, "_apply_rebalance_pipeline", fake_apply)

    original_clip = sim_runner.pd.Series.clip
    clip_calls: list[tuple[float | None, float | None]] = []

    def tracking_clip(self, lower=None, upper=None, *args, **kwargs):
        clip_calls.append((lower, upper))
        return original_clip(self, lower=lower, upper=upper, *args, **kwargs)

    monkeypatch.setattr(sim_runner.pd.Series, "clip", tracking_clip)

    policy = PolicyConfig(top_k=2, max_weight=0.4, min_track_months=0)
    sim.run(
        start=index[0],
        end=index[-1],
        freq="ME",
        lookback_months=1,
        policy=policy,
    )

    assert clip_calls, "Expected weight clipping to be attempted"
    assert any(upper == policy.max_weight for _, upper in clip_calls)
    assert captured_targets and list(captured_targets[0].index) == ["A", "B"]


def test_simulator_handles_equity_curve_update_failure(monkeypatch, caplog):
    import logging

    periods = pd.period_range("2020-01", "2020-02", freq="M")
    index = periods.to_timestamp(how="end")
    data = pd.DataFrame({"A": [0.1, 0.2]}, index=index)
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
        start=index[0],
        end=index[0],
        freq="ME",
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


def test_apply_rebalance_pipeline_periodic_skip_and_min_trade():
    prev = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.62, "B": 0.38})
    rb_state = {"since_last_reb": 0}
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["periodic_rebalance", "drift_band"],
        "params": {
            "periodic_rebalance": {"interval": 3},
            "drift_band": {"band_pct": 0.01, "min_trade": 0.5, "mode": "partial"},
        },
    }
    policy = PolicyConfig(max_weight=1.0)

    res = sim_runner._apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )

    # Periodic rebalance should defer because interval not yet met
    assert rb_state["since_last_reb"] == 1
    # Drift adjustments below the minimum trade should leave weights unchanged
    pd.testing.assert_series_equal(res.sort_index(), prev.sort_index().astype(float))


def test_apply_rebalance_pipeline_vol_and_drawdown(monkeypatch):
    prev = pd.Series({"A": 0.6, "B": 0.4})
    target = pd.Series({"A": 0.2, "B": 0.8})
    rb_state = {"since_last_reb": 0, "equity_curve": [1.0, 1.2, 0.9], "guard_on": False}
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drift_band", "vol_target_rebalance", "drawdown_guard"],
        "params": {
            "drift_band": {"band_pct": 0.0, "min_trade": 0.0, "mode": "full"},
            "vol_target_rebalance": {
                "target": 0.10,
                "window": 2,
                "lev_min": 0.5,
                "lev_max": 2.0,
            },
            "drawdown_guard": {
                "dd_window": 3,
                "dd_threshold": 0.05,
                "guard_multiplier": 0.5,
                "recover_threshold": 0.02,
            },
        },
    }
    policy = PolicyConfig(max_weight=1.0)

    res1 = sim_runner._apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )

    assert isinstance(res1, pd.Series)
    assert rb_state["guard_on"] is True

    # Simulate recovery to toggle guard off while running through strategies again
    rb_state["equity_curve"] = [1.0, 1.2, 1.18]
    rb_state["guard_on"] = True
    res2 = sim_runner._apply_rebalance_pipeline(
        prev_weights=res1,
        target_weights=target,
        date=pd.Timestamp("2020-02-29"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=policy,
    )

    assert isinstance(res2, pd.Series)
    assert rb_state["guard_on"] is False


def test_simulator_equity_curve_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        sim_runner, "compute_score_frame", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(
        sim_runner, "decide_hires_fires", lambda *a, **k: {"hire": [], "fire": []}
    )

    def bad_rebalance(prev_weights, target_weights, date, rb_cfg, rb_state, policy):
        rb_state["equity_curve"] = [1.0, "bad"]
        return target_weights

    monkeypatch.setattr(sim_runner, "_apply_rebalance_pipeline", bad_rebalance)
    calls: list[tuple] = []
    monkeypatch.setattr(sim_runner.logger, "warning", lambda *a, **k: calls.append(a))
    index = pd.period_range("2020-01", "2020-03", freq="M").to_timestamp(how="end")
    data = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=index)
    sim = sim_runner.Simulator(data)
    policy = PolicyConfig()
    sim.run(
        start=index[0],
        end=index[1],
        freq="ME",
        lookback_months=1,
        policy=policy,
    )
    assert any("Failed to update equity curve" in str(msg[0]) for msg in calls)


def test_simulator_run_handles_missing_forward_month(monkeypatch):
    data = pd.DataFrame({"A": [0.1]}, index=[pd.Timestamp("2020-01-31")])
    sim = sim_runner.Simulator(data)

    monkeypatch.setattr(
        sim_runner,
        "compute_score_frame",
        lambda *a, **k: pd.DataFrame({"m": [0.5]}, index=["A"]),
    )
    monkeypatch.setattr(
        sim_runner,
        "decide_hires_fires",
        lambda *a, **k: {"hire": [], "fire": []},
    )

    def passthrough_rebalance(*, target_weights, **_):
        return target_weights

    monkeypatch.setattr(sim_runner, "_apply_rebalance_pipeline", passthrough_rebalance)

    policy = PolicyConfig(top_k=0, bottom_k=0, min_track_months=0)

    result = sim.run(
        start=pd.Timestamp("2020-01-31"),
        end=pd.Timestamp("2020-01-31"),
        freq="ME",
        lookback_months=1,
        policy=policy,
    )

    assert result.portfolio.empty
    assert list(result.weights.keys()) == [
        pd.Timestamp("2020-01-31 23:59:59.999999999")
    ]
