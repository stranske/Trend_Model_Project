import pandas as pd
import pytest
from trend_portfolio_app.policy_engine import PolicyConfig
from trend_portfolio_app.sim_runner import _apply_rebalance_pipeline


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
        "params": {"drift_band": {"band_pct": 0.03, "min_trade": 0.001, "mode": "partial"}},
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


def test_turnover_cap_limits_moves():
    prev = pd.Series({"A": 0.50, "B": 0.50}, dtype=float)
    target = pd.Series({"A": 0.80, "B": 0.20}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["turnover_cap"],
        "params": {"turnover_cap": {"max_turnover": 0.10, "priority": "largest_gap"}},
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
    # Sum of absolute trades should be close to 0.10
    trn = float((out - prev).abs().sum())
    assert trn == pytest.approx(0.10, abs=1e-6)
    # Move should be in the correct direction
    assert out["A"] > prev["A"]
    assert out["B"] < prev["B"]


def test_vol_target_scales_within_bounds():
    prev = pd.Series({"A": 0.50, "B": 0.50}, dtype=float)
    target = pd.Series({"A": 0.50, "B": 0.50}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["vol_target_rebalance"],
        "params": {
            "vol_target_rebalance": {
                "target": 0.10,
                "window": 6,
                "lev_min": 0.8,
                "lev_max": 1.2,
            }
        },
    }
    # Craft an equity curve with high realized vol so lev factor clamps to lev_min
    ec = [1.0]
    for r in [0.05, -0.04, 0.06, -0.05, 0.04, -0.03, 0.05]:
        ec.append(ec[-1] * (1.0 + r))
    rb_state = {"equity_curve": ec}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-07-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    # Gross should be within lev bounds [0.8, 1.2]
    gross = float(out.sum())
    assert 0.8 - 1e-6 <= gross <= 1.2 + 1e-6


def test_drawdown_guard_reduces_exposure_on_dd():
    prev = pd.Series({"A": 0.60, "B": 0.40}, dtype=float)
    target = pd.Series({"A": 0.60, "B": 0.40}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": ["drawdown_guard"],
        "params": {
            "drawdown_guard": {
                "dd_window": 5,
                "dd_threshold": 0.10,
                "guard_multiplier": 0.5,
                "recover_threshold": 0.05,
            }
        },
    }
    # Equity curve experiencing a ~-15% drawdown
    ec = [1.0, 1.02, 0.98, 0.95, 0.90, 0.88]
    rb_state = {"equity_curve": ec}
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-06-30"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )
    # Exposure should be reduced by guard_multiplier (approx half gross)
    assert float(out.sum()) == pytest.approx(0.5, abs=1e-6)


def test_combined_rebalance_branches():
    prev = pd.Series({"A": 0.6, "B": 0.4}, dtype=float)
    target = pd.Series({"A": 0.2, "B": 0.8}, dtype=float)
    rb_cfg = {
        "bayesian_only": False,
        "strategies": [
            "drift_band",
            "turnover_cap",
            "vol_target_rebalance",
            "drawdown_guard",
        ],
        "params": {
            "drift_band": {"band_pct": 0.01, "min_trade": 0.001, "mode": "partial"},
            "turnover_cap": {"max_turnover": 0.05},
            "vol_target_rebalance": {
                "target": 0.2,
                "window": 3,
                "lev_min": 0.5,
                "lev_max": 1.5,
            },
            "drawdown_guard": {
                "dd_window": 4,
                "dd_threshold": 0.05,
                "guard_multiplier": 0.4,
                "recover_threshold": 0.02,
            },
        },
    }
    rb_state = {
        "equity_curve": [1.0, 1.1, 0.95, 1.05, 0.90, 0.85],
        "since_last_reb": 0,
    }
    out = _apply_rebalance_pipeline(
        prev_weights=prev,
        target_weights=target,
        date=pd.Timestamp("2020-07-31"),
        rb_cfg=rb_cfg,
        rb_state=rb_state,
        policy=_pc(),
    )

    assert rb_state.get("guard_on") is True
    assert out.index.tolist() == ["A", "B"]
    assert out.sum() == pytest.approx(out.clip(lower=0).sum(), abs=1e-9)
