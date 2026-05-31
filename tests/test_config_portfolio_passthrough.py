"""Regression: the config loader must NOT drop portfolio sub-keys.

PortfolioSettings (config/model.py) historically used model_config
extra="ignore", which silently discarded selection_mode/rank/selector/
custom_weights/weighting/constraints from any loaded YAML config. The engine
then fell back to selection_mode="all" and ignored the configured selection,
weighting, and constraints entirely.

These tests pin the pass-through so the regression cannot recur.
"""

from __future__ import annotations

from trend_analysis.config import load, load_config

# Keys the engine reads from portfolio but that are not declared fields on
# PortfolioSettings; all must survive a load.
_PASSTHROUGH_KEYS = (
    "selection_mode",
    "rank",
    "selector",
    "custom_weights",
    "weighting",
)


def test_demo_config_preserves_portfolio_selection_fields():
    cfg = load("config/demo.yml")
    portfolio = cfg.portfolio
    for key in _PASSTHROUGH_KEYS:
        assert key in portfolio, f"loader dropped portfolio.{key}"
    assert portfolio.get("selection_mode") == "rank"
    assert (portfolio.get("rank") or {}).get("n") == 5


def test_loader_preserves_arbitrary_portfolio_constraints():
    """A constraints block (incl. max_weight) must reach the loaded config.

    Built on the valid demo config so all required sections are present; we add a
    constraints block + weighting_scheme that the loader previously dropped.
    """
    from pathlib import Path

    import yaml

    with open("config/demo.yml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # load_config(dict) resolves csv_path against the repo root, so make it absolute
    # (demo's "../demo/..." is relative to the config/ dir).
    raw["data"]["csv_path"] = str((Path("config") / raw["data"]["csv_path"]).resolve())
    raw["portfolio"]["constraints"] = {"long_only": True, "max_weight": 0.2}
    raw["portfolio"]["weighting_scheme"] = "risk_parity"

    cfg = load_config(raw)
    portfolio = cfg.portfolio
    assert (portfolio.get("constraints") or {}).get("max_weight") == 0.2
    assert portfolio.get("weighting_scheme") == "risk_parity"
