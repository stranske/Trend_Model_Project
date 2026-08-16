from __future__ import annotations

import pytest

from trend_analysis.multi_period.engine import _resolve_portfolio_weighting
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


def test_weighting_config_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="portfolio.weighting must be a mapping"):
        _resolve_portfolio_weighting({"weighting": []})
