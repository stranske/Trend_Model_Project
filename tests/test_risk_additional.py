import numpy as np
import pandas as pd
import pytest

import trend_analysis.risk as risk


@pytest.fixture(autouse=True)
def _restore_optimizer(monkeypatch):
    original = risk.optimizer_mod.apply_constraints
    yield
    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", original)


def test_realised_volatility_supports_simple_and_ewma():
    returns = pd.DataFrame(
        {"asset": [0.01, 0.03, 0.02, 0.05, 0.04]},
        index=pd.date_range("2024-01-01", periods=5, freq="M"),
    )
    simple = risk.realised_volatility(returns, risk.RiskWindow(length=2))
    assert simple.index.equals(returns.index)
    assert simple.columns.tolist() == ["asset"]
    assert np.isfinite(simple.iloc[-1, 0])

    ewma = risk.realised_volatility(
        returns,
        risk.RiskWindow(length=2, decay="ewma", ewma_lambda=0.8),
    )
    assert ewma.index.equals(returns.index)
    assert np.isfinite(ewma.iloc[-1, 0])


def test_realised_volatility_validates_inputs():
    with pytest.raises(ValueError, match="returns cannot be empty"):
        risk.realised_volatility(pd.DataFrame(), risk.RiskWindow(length=2))

    with pytest.raises(ValueError, match="window length must be positive"):
        risk.realised_volatility(pd.DataFrame([[0.1]]), risk.RiskWindow(length=0))

    with pytest.raises(ValueError):
        risk.realised_volatility(
            pd.DataFrame([[0.1], [0.2]]),
            risk.RiskWindow(length=2, decay="ewma", ewma_lambda=1.5),
        )


def test_compute_constrained_weights_applies_controls(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.015, 0.018, 0.022],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-31", periods=5, freq="M"),
    )
    base_weights = {"A": 0.6, "B": 0.4}
    payload_capture: dict[str, object] = {}

    def fake_apply_constraints(weights: pd.Series, payload: dict[str, object]):
        payload_capture["weights"] = weights.copy()
        payload_capture["payload"] = payload.copy()
        return pd.Series({"A": 0.8, "B": 0.2})

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", fake_apply_constraints)

    result, diagnostics = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=risk.RiskWindow(length=3),
        target_vol=0.15,
        periods_per_year=12.0,
        floor_vol=0.05,
        long_only=True,
        max_weight=0.7,
        previous_weights={"A": 0.1, "B": 0.9},
        max_turnover=0.4,
        group_caps={"grp": 1.0},
        groups={"A": "grp", "B": "grp"},
    )

    assert pytest.approx(result.sum()) == 1.0
    assert diagnostics.turnover_value <= 0.4 + 1e-9
    assert diagnostics.turnover.index[-1] == returns.index[-1]
    assert diagnostics.asset_volatility.shape == returns.shape
    assert diagnostics.scale_factors.index.tolist() == result.index.tolist()
    assert payload_capture["payload"]["long_only"] is True
    assert payload_capture["payload"]["max_weight"] == 0.7
    assert payload_capture["payload"]["group_caps"] == {"grp": 1.0}
    assert payload_capture["payload"]["groups"] == {"A": "grp", "B": "grp"}


def test_compute_constrained_weights_handles_non_datetime_index(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.02, 0.01, -0.03],
            "B": [0.01, -0.02, 0.04],
        }
    )

    def passthrough(weights: pd.Series, payload: dict[str, object]):
        return weights

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", passthrough)

    result, diagnostics = risk.compute_constrained_weights(
        pd.Series({"A": 1.0, "B": 2.0}),
        returns,
        window=risk.RiskWindow(length=2),
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=False,
        max_weight=None,
        previous_weights=None,
        max_turnover=None,
        group_caps=None,
        groups=None,
    )

    assert diagnostics.turnover.index.tolist() == [pd.Timestamp("1970-01-01")]
    assert pytest.approx(result.sum()) == 1.0


def test_compute_constrained_weights_validates_inputs(monkeypatch):
    returns = pd.DataFrame([[0.01, 0.02]])

    with pytest.raises(ValueError, match="returns cannot be empty"):
        risk.compute_constrained_weights(
            {"A": 1.0},
            returns.iloc[0:0],
            window=risk.RiskWindow(length=2),
            target_vol=0.1,
            periods_per_year=12.0,
            floor_vol=None,
            long_only=True,
            max_weight=None,
        )

    with pytest.raises(ValueError, match="base_weights cannot be empty"):
        risk.compute_constrained_weights(
            {},
            returns,
            window=risk.RiskWindow(length=2),
            target_vol=0.1,
            periods_per_year=12.0,
            floor_vol=None,
            long_only=True,
            max_weight=None,
        )


def test_periods_per_year_from_code_defaults():
    assert risk.periods_per_year_from_code(None) == 12.0
    assert risk.periods_per_year_from_code("w") == 52.0
    assert risk.periods_per_year_from_code("unknown") == 12.0
