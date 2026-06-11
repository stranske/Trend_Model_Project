from __future__ import annotations

from trend_analysis.config import load
from trend_analysis.multi_period import run_from_config

from .harness import REPO_ROOT, apply_patch


def test_multi_period_max_funds_caps_realised_holdings_per_period() -> None:
    cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(cfg, {"multi_period.max_funds": 5})

    results = run_from_config(cfg)

    assert results
    for result in results:
        fund_weights = result.get("fund_weights")
        assert isinstance(fund_weights, dict)
        selected = [
            fund for fund, weight in fund_weights.items() if abs(float(weight)) > 1e-9
        ]
        assert len(selected) <= 5


def test_portfolio_rank_n_controls_selected_fund_count_per_period() -> None:
    control_cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(control_cfg, {"portfolio.rank.n": 10})
    variant_cfg = load(str(REPO_ROOT / "config/demo.yml"))
    apply_patch(variant_cfg, {"portfolio.rank.n": 5})

    control_counts = [
        len(result["selected_funds"]) for result in run_from_config(control_cfg)
    ]
    variant_counts = [
        len(result["selected_funds"]) for result in run_from_config(variant_cfg)
    ]

    assert control_counts
    assert len(control_counts) == len(variant_counts)
    assert all(count == 10 for count in control_counts)
    assert all(count == 5 for count in variant_counts)
