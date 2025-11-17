from pathlib import Path

import pytest
import yaml

from trend_analysis.config import Config


def load_defaults():
    return yaml.safe_load(Path("config/defaults.yml").read_text())


def make_cfg(overrides):
    data = load_defaults()
    # Ensure required minimal keys for constructing Config remain as in defaults
    for k, v in overrides.items():
        # support nested portfolio updates
        if k == "portfolio":
            data.setdefault("portfolio", {}).update(v)
        else:
            data[k] = v
    return data


@pytest.mark.parametrize("tc", [0, 5, 12.5, 0.0])
def test_transaction_cost_bps_valid(tc):
    cfg_dict = make_cfg({"portfolio": {"transaction_cost_bps": tc}})
    cfg = Config(**cfg_dict)
    assert float(cfg.portfolio.get("transaction_cost_bps")) == float(tc)


@pytest.mark.parametrize("tc", [-1, -0.01])
def test_transaction_cost_bps_invalid(tc):
    cfg_dict = make_cfg({"portfolio": {"transaction_cost_bps": tc}})
    with pytest.raises(Exception):
        Config(**cfg_dict)


@pytest.mark.parametrize("cap", [0, 0.1, 0.5, 1.0, 1.5, 2.0])
def test_max_turnover_valid(cap):
    cfg_dict = make_cfg({"portfolio": {"max_turnover": cap}})
    cfg = Config(**cfg_dict)
    assert float(cfg.portfolio.get("max_turnover")) == float(cap)


@pytest.mark.parametrize("cap", [-0.1, -1, 2.01, 5])
def test_max_turnover_invalid(cap):
    cfg_dict = make_cfg({"portfolio": {"max_turnover": cap}})
    with pytest.raises(Exception):
        Config(**cfg_dict)


def test_string_coercion():
    cfg_dict = make_cfg(
        {"portfolio": {"transaction_cost_bps": "15", "max_turnover": "0.75"}}
    )
    cfg = Config(**cfg_dict)
    assert cfg.portfolio["transaction_cost_bps"] == 15.0
    assert cfg.portfolio["max_turnover"] == 0.75
@pytest.mark.parametrize("slip", [0, 2.5, 15.0])
def test_slippage_bps_valid(slip):
    cfg_dict = make_cfg({"portfolio": {"slippage_bps": slip}})
    cfg = Config(**cfg_dict)
    assert float(cfg.portfolio.get("slippage_bps")) == float(slip)


@pytest.mark.parametrize("slip", [-0.01, -5])
def test_slippage_bps_invalid(slip):
    cfg_dict = make_cfg({"portfolio": {"slippage_bps": slip}})
    with pytest.raises(Exception):
        Config(**cfg_dict)

