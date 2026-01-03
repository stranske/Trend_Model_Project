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
        index=pd.date_range("2024-01-01", periods=5, freq="ME"),
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
        index=pd.date_range("2024-01-31", periods=5, freq="ME"),
    )
    base_weights = {"A": 1.2, "B": 0.8}
    payload_capture: dict[str, object] = {}
    captured_calls: list[dict[str, object]] = []

    def fake_apply_constraints(weights: pd.Series, payload: dict[str, object]):
        captured_calls.append({"weights": weights.copy(), "payload": payload.copy()})
        payload_capture["weights"] = weights.copy()
        payload_capture["payload"] = payload.copy()
        return weights

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
    assert len(captured_calls) == 2
    assert pytest.approx(captured_calls[0]["weights"].sum()) == 1.0


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


def test_compute_constrained_weights_skips_scaling_when_disabled(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.2, -0.2],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=pd.date_range("2024-01-31", periods=5, freq="ME"),
    )

    def passthrough(weights: pd.Series, payload: dict[str, object]):
        return weights

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", passthrough)

    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    window = risk.RiskWindow(length=3, decay="simple")

    disabled, diag_disabled = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=window,
        target_vol=None,
        periods_per_year=12.0,
        floor_vol=0.0,
        long_only=True,
        max_weight=None,
    )
    enabled, _ = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=window,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=0.0,
        long_only=True,
        max_weight=None,
    )

    assert disabled.index.tolist() == ["A", "B"]
    assert disabled.loc["A"] == pytest.approx(0.5)
    assert disabled.loc["B"] == pytest.approx(0.5)
    assert not np.allclose(disabled.values, enabled.values)
    assert np.allclose(diag_disabled.scale_factors.values, 1.0)


def test_compute_constrained_weights_window_length_changes_scaling(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01, 0.2, -0.2],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )

    def passthrough(weights: pd.Series, payload: dict[str, object]):
        return weights

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", passthrough)

    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    short_window = risk.RiskWindow(length=2, decay="simple")
    long_window = risk.RiskWindow(length=5, decay="simple")

    _, diag_short = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=short_window,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )
    _, diag_long = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=long_window,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    assert diag_short.scale_factors.loc["A"] != diag_long.scale_factors.loc["A"]


def test_compute_constrained_weights_decay_changes_scaling(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01, 0.2, -0.2],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )

    def passthrough(weights: pd.Series, payload: dict[str, object]):
        return weights

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", passthrough)

    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    simple_window = risk.RiskWindow(length=4, decay="simple")
    ewma_window = risk.RiskWindow(length=4, decay="ewma", ewma_lambda=0.5)

    _, diag_simple = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=simple_window,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )
    _, diag_ewma = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=ewma_window,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    assert diag_simple.scale_factors.loc["A"] != diag_ewma.scale_factors.loc["A"]


def test_compute_constrained_weights_ewma_lambda_changes_scaling(monkeypatch):
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01, 0.2, -0.2],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )

    def passthrough(weights: pd.Series, payload: dict[str, object]):
        return weights

    monkeypatch.setattr(risk.optimizer_mod, "apply_constraints", passthrough)

    base_weights = pd.Series({"A": 0.5, "B": 0.5})
    fast_decay = risk.RiskWindow(length=4, decay="ewma", ewma_lambda=0.5)
    slow_decay = risk.RiskWindow(length=4, decay="ewma", ewma_lambda=0.9)

    _, diag_fast = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=fast_decay,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )
    _, diag_slow = risk.compute_constrained_weights(
        base_weights,
        returns,
        window=slow_decay,
        target_vol=0.2,
        periods_per_year=12.0,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    assert diag_fast.scale_factors.loc["A"] != diag_slow.scale_factors.loc["A"]


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
