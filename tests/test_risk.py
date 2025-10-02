import numpy as np
import pandas as pd

from trend_analysis.risk import RiskConfig, apply_risk_controls


def _make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    data = {
        "A": [0.02, 0.01, 0.03, 0.04, 0.02, 0.01],
        "B": [0.01, -0.02, 0.02, 0.03, 0.01, 0.00],
    }
    return pd.DataFrame(data, index=dates)


def test_apply_risk_controls_enforces_constraints():
    returns = _make_returns()
    in_returns = returns.iloc[:3]
    out_returns = returns.iloc[3:]

    base_equal = pd.Series([0.5, 0.5], index=["A", "B"], dtype=float)
    base_user = pd.Series([0.8, -0.2], index=["A", "B"], dtype=float)
    prev_user = pd.Series([0.4, 0.6], index=["A", "B"], dtype=float)

    cfg = RiskConfig(
        target_vol=0.1,
        floor_vol=0.02,
        warmup_periods=1,
        lookback=2,
        annualisation=12.0,
        long_only=True,
        max_weight=0.7,
        turnover_cap=0.2,
        transaction_cost=0.0,
    )

    result = apply_risk_controls(
        in_returns=in_returns,
        out_returns=out_returns,
        base_equal_weights=base_equal,
        base_user_weights=base_user,
        cfg=cfg,
        prev_user_weights=prev_user,
    )

    assert np.isclose(result.equal_weights.sum(), 1.0)
    assert np.isclose(result.user_weights.sum(), 1.0)
    assert (result.user_weights >= 0).all()

    turnover = result.diagnostics.turnover
    if not turnover.empty:
        assert float(turnover["user_weight"].iloc[0]) <= 0.2 + 1e-6

    realized = result.diagnostics.realized_vol
    assert isinstance(realized, pd.DataFrame)
    assert not realized.empty


def test_apply_risk_controls_respects_floor_and_scale():
    returns = _make_returns()
    in_returns = returns.iloc[:4]
    out_returns = returns.iloc[4:]
    base_weights = pd.Series([0.6, 0.4], index=["A", "B"], dtype=float)

    cfg = RiskConfig(
        target_vol=0.2,
        floor_vol=0.05,
        warmup_periods=0,
        lookback=3,
        annualisation=12.0,
        long_only=False,
        max_weight=None,
        turnover_cap=None,
        transaction_cost=0.0,
    )

    result = apply_risk_controls(
        in_returns=in_returns,
        out_returns=out_returns,
        base_equal_weights=base_weights,
        base_user_weights=base_weights,
        cfg=cfg,
    )

    scale = result.diagnostics.scale_factors
    assert (scale >= 0).all()
    assert not scale.isna().any()

    # Floor should prevent explosive scale on near-zero volatility asset
    assert float(scale.max()) <= (cfg.target_vol / cfg.floor_vol) + 1e-6
