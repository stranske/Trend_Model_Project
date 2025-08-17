import pytest
import pandas as pd
from trend_portfolio_app.sim_runner import _apply_rebalance_pipeline
from trend_portfolio_app.policy_engine import PolicyConfig


def _pc() -> PolicyConfig:
    return PolicyConfig(max_weight=1.0)


def test_bayesian_only_passthrough():
    prev = pd.Series({"A": 0.6, "B": 0.4}, dtype=float)
    target = pd.Series({"A": 0.2, "B": 0.8}, dtype=float)
    rb_cfg = {"bayesian_only": True}
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    # Should equal target
    pd.testing.assert_series_equal(out.sort_index(), target.sort_index())


def test_periodic_rebalance_every_two():
    prev = pd.Series({"A": 0.5, "B": 0.5}, dtype=float)
    target = pd.Series({"A": 0.0, "B": 1.0}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["periodic_rebalance"],
        "params": {"periodic_rebalance": {"interval": 2}},
    }
    rb_state = {"since_last_reb": 0}
    # First call: interval condition not met yet, so no rebalance
    out1 = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    pd.testing.assert_series_equal(out1.sort_index(), prev.sort_index())
    # Second call: should now rebalance to target
    out2 = _apply_rebalance_pipeline(
        prev_weights=out1,
        target_weights=target,
        date=pd.Timestamp("2020-02-29"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    pd.testing.assert_series_equal(out2.sort_index(), target.sort_index())


def test_drift_band_partial_adjusts():
    prev = pd.Series({"A": 0.50, "B": 0.50}, dtype=float)
    target = pd.Series({"A": 0.60, "B": 0.40}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drift_band"],
        "params": {
            "drift_band": {"band_pct": 0.03, "min_trade": 0.001, "mode": "partial"}
        },
    }
    rb_state = {}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-01-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    assert out["A"] > prev["A"] and out["A"] < target["A"]
    assert pytest.approx(out.sum(), rel=0, abs=1e-9) == 1.0
