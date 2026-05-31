"""Parameter-consistency / feasibility validation (finding #4).

A simulation must not run on internally contradictory or infeasible parameters.
The only escape for a capacity shortfall is an EXPLICIT cash allocation.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from trend_analysis.config.validation import validate_config


@pytest.fixture(scope="module")
def demo_cfg() -> dict:
    with open("config/demo.yml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _errors(cfg: dict) -> list[str]:
    result = validate_config(cfg, skip_required_fields=True)
    return [e.message for e in result.errors]


def test_demo_config_is_feasible(demo_cfg):
    assert not [m for m in _errors(demo_cfg) if "Infeasible" in m or "below min_weight" in m]


def test_infeasible_max_weight_is_rejected(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)  # selection_mode=rank, rank.n=5
    cfg["portfolio"]["constraints"] = {"long_only": True, "max_weight": 0.1}  # 0.1*5 = 0.5 < 1
    assert any("Infeasible" in m and "max_weight" in m for m in _errors(cfg))


def test_explicit_cash_allows_otherwise_infeasible_cap(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)
    cfg["portfolio"]["constraints"] = {"long_only": True, "max_weight": 0.1, "cash_weight": 0.6}
    assert not [m for m in _errors(cfg) if "Infeasible" in m]


def test_max_weight_below_min_weight_is_rejected(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)
    cfg["portfolio"]["constraints"] = {"max_weight": 0.1, "min_weight": 0.3}
    assert any("below min_weight" in m for m in _errors(cfg))


def test_floor_vol_not_below_target_is_rejected(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)
    cfg["vol_adjust"] = {"enabled": True, "floor_vol": 0.2, "target_vol": 0.1}
    assert any("floor_vol" in m for m in _errors(cfg))


def test_min_funds_exceeds_max_funds_is_rejected(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)
    cfg["multi_period"] = {**(cfg.get("multi_period") or {}), "min_funds": 9, "max_funds": 5}
    assert any("min_funds exceeds max_funds" in m for m in _errors(cfg))
