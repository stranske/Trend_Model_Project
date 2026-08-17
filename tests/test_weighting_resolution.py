from __future__ import annotations

from types import SimpleNamespace

import pytest
import pandas as pd

from trend_analysis.multi_period.engine import (
    _portfolio_weighting_config,
    _resolve_portfolio_weighting,
)
from trend_analysis.plugins import WeightEngine, weight_engine_registry
from trend_analysis.weighting import EqualWeight, ScorePropSimple
from trend_analysis.weights.risk_parity import RiskParity


def test_weighting_name_risk_parity_uses_risk_engine() -> None:
    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {"weighting": {"name": "risk_parity"}}
    )

    assert scheme == "risk_parity"
    assert use_risk is True
    assert isinstance(risk_engine, RiskParity)
    assert fallback is None
    assert isinstance(weighting, EqualWeight)


def test_custom_weighting_is_a_non_plugin_mode() -> None:
    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {"weighting": {"name": "custom"}, "custom_weights": {"FundA": 60, "FundB": 40}}
    )

    assert scheme == "custom"
    assert use_risk is False
    assert risk_engine is None
    assert fallback is None
    assert isinstance(weighting, EqualWeight)


def test_weighting_name_score_prop_is_reachable() -> None:
    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {"weighting": {"name": "score_prop", "params": {"column": "Sortino"}}}
    )

    assert scheme == "score_prop"
    assert isinstance(weighting, ScorePropSimple)
    assert weighting.column == "Sortino"
    assert use_risk is False
    assert risk_engine is None
    assert fallback is None


def test_nested_bayesian_weighting_name_is_preserved() -> None:
    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {"weighting": {"name": "score_prop_bayes", "params": {"column": "Sharpe"}}}
    )

    assert scheme == "score_prop_bayes"
    assert weighting.__class__.__name__ == "ScorePropBayesian"
    assert use_risk is False
    assert risk_engine is None
    assert fallback is None


def test_nested_score_weighting_name_is_preserved() -> None:
    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {"weighting": {"name": "score_prop", "params": {"column": "Sortino"}}}
    )

    assert scheme == "score_prop"
    assert isinstance(weighting, ScorePropSimple)
    assert use_risk is False
    assert risk_engine is None
    assert fallback is None


def test_unknown_weighting_raises() -> None:
    with pytest.raises(ValueError, match="ew.*robust"):
        _resolve_portfolio_weighting({"weighting": {"name": "not_a_scheme"}})


def test_registered_third_party_weighting_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class ThirdPartyWeighting(WeightEngine):
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def weight(self, cov: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0 / len(cov), index=cov.index)

    monkeypatch.setitem(
        weight_engine_registry._plugins,
        "third_party_weight_engine",
        ThirdPartyWeighting,
    )

    weighting, use_risk, risk_engine, fallback, scheme = _resolve_portfolio_weighting(
        {
            "weighting": {
                "name": "third_party_weight_engine",
                "params": {"scale": 2.5},
            }
        }
    )

    assert scheme == "third_party_weight_engine"
    assert use_risk is True
    assert isinstance(risk_engine, ThirdPartyWeighting)
    assert risk_engine.scale == pytest.approx(2.5)
    assert fallback is None
    assert isinstance(weighting, EqualWeight)


def test_weighting_config_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="portfolio.weighting must be a mapping"):
        _resolve_portfolio_weighting({"weighting": []})


def test_declared_root_robustness_reaches_multi_period_weighting() -> None:
    robustness = {"condition_check": {"enabled": True, "threshold": 10.0}}
    cfg = SimpleNamespace(
        portfolio={"weighting": {"name": "robust_mv"}},
        robustness=robustness,
    )

    assert _portfolio_weighting_config(cfg)["robustness"] == robustness


def test_nested_robustness_takes_precedence_over_declared_root_section() -> None:
    nested = {"condition_check": {"enabled": False}}
    cfg = SimpleNamespace(
        portfolio={"weighting": {"name": "robust_mv"}, "robustness": nested},
        robustness={"condition_check": {"enabled": True}},
    )

    assert _portfolio_weighting_config(cfg)["robustness"] == nested
