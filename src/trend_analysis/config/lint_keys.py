"""Config key linting for sections that intentionally allow extra keys."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from utils.paths import proj_path

_DEFAULTS_PATH = proj_path() / "config" / "defaults.yml"

_DECLARED_PORTFOLIO_KEYS = {
    "rebalance_calendar",
    "rebalance_freq",
    "max_turnover",
    "transaction_cost_bps",
    "lambda_tc",
    "min_tenure_n",
    "min_tenure_periods",
    "ci_level",
    "cost_model",
    "cost_model.bps_per_trade",
    "cost_model.slippage_bps",
    "cost_model.per_trade_bps",
    "cost_model.half_spread_bps",
    "turnover_cap",
    "weight_policy",
    "cooldown_periods",
    "cooldown_months",
}

_CONSUMED_PORTFOLIO_KEYS = {
    "policy",
    "selection_mode",
    "target_n",
    "entry_soft_strikes",
    "entry_eligible_strikes",
    "random_n",
    "manual_list",
    "indices_list",
    "weighting_scheme",
    "leverage_cap",
    "rank",
    "rank.inclusion_approach",
    "rank.n",
    "rank.pct",
    "rank.threshold",
    "rank.bottom_k",
    "rank.score_by",
    "rank.transform",
    "rank.limit_one_per_firm",
    "rank.blended_weights",
    "selector",
    "selector.name",
    "selector.params",
    "weighting",
    "weighting.name",
    "weighting.params",
    "constraints",
    "constraints.long_only",
    "constraints.max_funds",
    "constraints.max_weight",
    "constraints.min_weight",
    "constraints.max_active_positions",
    "constraints.group_caps",
    "constraints.cash_weight",
    "constraints.allowed_assets",
    "robustness",
    "robustness.shrinkage",
    "robustness.shrinkage.enabled",
    "robustness.shrinkage.method",
    "robustness.condition_check",
    "robustness.condition_check.enabled",
    "robustness.condition_check.threshold",
    "robustness.condition_check.safe_mode",
    "robustness.condition_check.diagonal_loading_factor",
    "robustness.logging",
    "robustness.logging.log_method_switches",
    "robustness.logging.log_shrinkage_intensity",
    "robustness.logging.log_condition_numbers",
}

_DYNAMIC_PORTFOLIO_SUBTREES = {
    "custom_weights",
    "rank",
    "rank.blended_weights",
    "selector.params",
    "threshold_hold",
    "weighting.params",
    "constraints.group_caps",
    "constraints.allowed_assets",
}


def lint_portfolio_keys(
    config: Mapping[str, Any],
    *,
    defaults_path: Path | None = None,
) -> list[str]:
    """Return unexpected ``portfolio.*`` key paths.

    ``PortfolioSettings`` keeps ``extra="allow"`` because the engine consumes
    plugin-like portfolio options that are not Pydantic fields. This lint keeps
    those known consumed keys explicit while rejecting misspelled or inert keys.
    """

    defaults = _load_defaults(defaults_path or _DEFAULTS_PATH)
    allowed = _allowed_portfolio_paths()
    unknown: set[str] = set()

    defaults_portfolio = defaults.get("portfolio")
    if isinstance(defaults_portfolio, Mapping):
        unknown.update(_unknown_portfolio_paths(defaults_portfolio, allowed))

    portfolio = config.get("portfolio")
    if isinstance(portfolio, Mapping):
        unknown.update(_unknown_portfolio_paths(portfolio, allowed))

    return sorted(f"portfolio.{path}" for path in unknown)


@lru_cache(maxsize=None)
def _load_defaults(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _allowed_portfolio_paths() -> set[str]:
    return set(_DECLARED_PORTFOLIO_KEYS) | set(_CONSUMED_PORTFOLIO_KEYS)


def _unknown_portfolio_paths(portfolio: Mapping[str, Any], allowed: set[str]) -> set[str]:
    return {
        path
        for path in _flatten_mapping_paths(portfolio)
        if not _is_allowed_portfolio_path(path, allowed)
    }


def _flatten_mapping_paths(mapping: Mapping[str, Any], *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for raw_key, value in mapping.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        paths.add(path)
        if isinstance(value, Mapping):
            paths.update(_flatten_mapping_paths(value, prefix=path))
    return paths


def _is_allowed_portfolio_path(path: str, allowed: set[str]) -> bool:
    if path in allowed:
        return True
    return any(
        path == subtree or path.startswith(f"{subtree}.")
        for subtree in _DYNAMIC_PORTFOLIO_SUBTREES
    )
