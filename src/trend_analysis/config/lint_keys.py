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

# Top-level config sections recognised by the engine. Sourced from the generated
# ``config.schema.json`` ``properties`` plus every top-level section shipped in
# ``config/*.yml`` (those keys are guarded as consumed by
# ``tests/config/test_no_inert_keys.py``). Unknown top-level sections are silent
# no-ops, so ``lint_config_sections`` rejects them (#5543, follow-up to A1/#5389).
_DECLARED_TOP_LEVEL_SECTIONS = {
    "benchmarks",
    "checkpoint_dir",
    "data",
    "export",
    "identity",
    "jobs",
    "metrics",
    "multi_period",
    "performance",
    "portfolio",
    "preprocessing",
    "regime",
    "run",
    "sample_split",
    "seed",
    "strategy",
    "version",
    "vol_adjust",
    "walk_forward",
}

# Closed-field key sets for the ``metrics``/``export``/``run`` sections, which the
# minimal ``TrendConfig`` does not model and therefore silently drops unknown keys
# under. Each set is the union of keys shipped in ``config/*.yml`` (consumed-key
# guarded by ``test_no_inert_keys.py``) plus the ``schema_generator`` declarations.
# Only immediate children are validated; nested mappings (registries, format
# lists, ...) carry user-supplied values and are left to their consumers.
_DECLARED_METRICS_KEYS = {"registry", "rf_override_enabled", "rf_rate_annual"}
_DECLARED_EXPORT_KEYS = {
    "directory",
    "disable_narrative_generation",
    "filename",
    "formats",
    "include_figures",
}
_DECLARED_RUN_KEYS = {
    "checkpoint_dir",
    "jobs",
    "monthly_cost",
    "n_jobs",
    "name",
    "output_dir",
    "seed",
}

_CLOSED_SECTIONS: dict[str, set[str]] = {
    "metrics": _DECLARED_METRICS_KEYS,
    "export": _DECLARED_EXPORT_KEYS,
    "run": _DECLARED_RUN_KEYS,
}


def lint_config_sections(config: Mapping[str, Any]) -> list[str]:
    """Return unexpected top-level sections and unknown closed-section keys.

    The minimal :class:`~trend_analysis.config.model.TrendConfig` only models
    ``data``/``portfolio``/``vol_adjust`` with ``extra="ignore"``, so unknown
    top-level sections and unknown keys under ``metrics``/``export``/``run`` load
    silently. This lint makes those fail loudly (#5543, follow-up to A1/#5389)
    while leaving the engine's free-form portfolio surface to
    :func:`lint_portfolio_keys`.

    Only immediate children of the closed sections are checked; nested mappings
    hold user-supplied values, not fixed schema keys.
    """

    unknown: list[str] = []
    for raw_key in config:
        key = str(raw_key)
        if key not in _DECLARED_TOP_LEVEL_SECTIONS:
            unknown.append(key)
    for section, allowed in _CLOSED_SECTIONS.items():
        block = config.get(section)
        if isinstance(block, Mapping):
            for raw_child in block:
                child = str(raw_child)
                if child not in allowed:
                    unknown.append(f"{section}.{child}")
    return sorted(unknown)


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
