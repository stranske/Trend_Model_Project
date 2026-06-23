from __future__ import annotations

import pandas as pd
from trend.diagnostics import DiagnosticResult

from trend_analysis.config import load
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.multi_period import run_from_config

from .harness import REPO_ROOT, ScenarioOutput, apply_patch


def test_multi_period_max_funds_caps_realised_holdings_per_period() -> None:
    cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(cfg, {"multi_period.max_funds": 5})

    results = run_from_config(cfg)

    assert results
    for result in results:
        fund_weights = result.get("fund_weights")
        assert isinstance(fund_weights, dict)
        selected = [fund for fund, weight in fund_weights.items() if abs(float(weight)) > 1e-9]
        assert len(selected) <= 5


def test_portfolio_rank_n_controls_selected_fund_count_per_period() -> None:
    control_cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(control_cfg, {"portfolio.rank.n": 10})
    variant_cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(variant_cfg, {"portfolio.rank.n": 5})

    control_counts = [len(result["selected_funds"]) for result in run_from_config(control_cfg)]
    variant_counts = [len(result["selected_funds"]) for result in run_from_config(variant_cfg)]

    assert control_counts
    assert len(control_counts) == len(variant_counts)
    assert all(count == 10 for count in control_counts)
    assert all(count == 5 for count in variant_counts)


def test_top_pct_rank_n_does_not_cap_percentage_selection() -> None:
    cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(
        cfg,
        {
            "portfolio.rank.inclusion_approach": "top_pct",
            "portfolio.rank.pct": 0.5,
            "portfolio.rank.n": 5,
            "portfolio.policy": "threshold_hold",
            "multi_period.max_funds": 50,
        },
    )

    selected_counts = [len(result["selected_funds"]) for result in run_from_config(cfg)]

    assert selected_counts
    assert all(count > 5 for count in selected_counts)


def test_pipeline_subset_manual_holdings_remain_realised_holdings(monkeypatch) -> None:
    def fake_pipeline(*_args, **kwargs):
        manual_funds = [str(fund) for fund in kwargs["manual_funds"]]
        kept = manual_funds[:3]
        weights = {fund: 1.0 / len(kept) for fund in kept}
        payload = {
            "fund_weights": weights,
            "out_sample_scaled": pd.DataFrame(
                {"portfolio": [0.01]},
                index=pd.date_range("2020-03-31", periods=1, freq="ME"),
            ),
            "score_frame": pd.DataFrame(index=manual_funds),
        }
        return DiagnosticResult(value=payload, diagnostic=None)

    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", fake_pipeline)

    cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(
        cfg,
        {
            "portfolio.policy": "threshold_hold",
            "portfolio.rank.n": 5,
            "multi_period.max_funds": 5,
        },
    )

    results = run_from_config(cfg)

    assert results
    for result in results:
        fund_weights = result.get("fund_weights")
        selected_funds = result.get("selected_funds")
        assert isinstance(fund_weights, dict)
        assert isinstance(selected_funds, list)
        assert len(selected_funds) == 3
        assert set(selected_funds) == set(fund_weights)


def test_num_selected_counts_nonzero_weights_when_no_selected_count() -> None:
    output = ScenarioOutput(
        metrics=pd.DataFrame(),
        weights=pd.Series({"A": 0.5, "B": 0.0}, dtype=float),
        fund_weights=pd.Series({"A": 0.5, "B": 0.0}, dtype=float),
        turnover=pd.Series(dtype=float),
        portfolio=pd.Series([0.01, 0.02], dtype=float),
        costs={},
        seed=42,
    )

    assert output.derived()["num_selected"] == 1
