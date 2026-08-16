"""Regression gates for the canonical missing-data configuration surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from trend.config_schema import CoreConfigError
from trend_analysis.config_contract import (
    resolve_portfolio_cost_bps,
)
from trend_analysis.multi_period.engine import _resolve_portfolio_weighting
from trend_analysis.signals import trend_spec_from_mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    "src/trend_analysis/data.py",
    "src/trend_analysis/tool_layer.py",
    "src/trend_analysis/pipeline_entrypoints.py",
    "src/trend_analysis/multi_period/engine.py",
    "src/trend_analysis/multi_period/loaders.py",
    "src/trend_analysis/config_contract.py",
    "src/trend_analysis/config/model.py",
    "src/trend_analysis/rebalancing/strategies.py",
)


def test_legacy_missing_policy_keys_are_not_read() -> None:
    """Runtime code must not revive the retired missing-data config aliases."""

    forbidden = ("nan" + "_policy", "nan" + "_limit")
    offenders = {
        relative_path: [key for key in forbidden if key in (REPO_ROOT / relative_path).read_text()]
        for relative_path in RUNTIME_PATHS
    }

    assert not {path: keys for path, keys in offenders.items() if keys}


def test_legacy_weighting_shape_is_rejected() -> None:
    with pytest.raises(CoreConfigError, match="portfolio.weighting.name"):
        _resolve_portfolio_weighting({"weighting_" + "scheme": "risk_parity"})


@pytest.mark.parametrize(
    "portfolio",
    [
        {"transaction_" + "cost_bps": 5},
        {"slippage_" + "bps": 2},
        {"cost_model": {"bps_per_" + "trade": 5}},
        {"cost_model": {"slippage_" + "bps": 2}},
    ],
)
def test_legacy_cost_shapes_are_rejected(portfolio: dict[str, object]) -> None:
    with pytest.raises(CoreConfigError, match="was removed"):
        resolve_portfolio_cost_bps(portfolio)


def test_multi_period_runtime_has_no_removed_shape_fallbacks() -> None:
    source = (REPO_ROOT / "src/trend_analysis/multi_period/engine.py").read_text()
    forbidden = (
        'get("weighting_' + 'scheme"',
        'get("sticky_drop_' + 'periods"',
        'get("max_' + 'active"',
        'get("min_tenure_' + 'periods"',
        "simplified " + "signature",
        "Some legacy configs use the " + "inverse",
    )

    assert not [token for token in forbidden if token in source]


def test_removed_rebalancing_alias_is_absent() -> None:
    source = (REPO_ROOT / "src/trend_analysis/rebalancing/strategies.py").read_text()
    assert "Rebalancing" + "Strategy" not in source


def test_signal_parser_has_no_removed_alias_reads() -> None:
    source = (REPO_ROOT / "src/trend_analysis/signals.py").read_text()
    forbidden = (
        "trend_" + "window",
        "trend_" + "lag",
        "trend_" + "min_periods",
        "trend_" + "vol_adjust",
        "trend_" + "vol_target",
    )
    assert not [token for token in forbidden if token in source]
    assert trend_spec_from_mapping({"trend_" + "zscore": 2.0}).zscore is False


def test_parallelism_has_no_top_level_alias_read() -> None:
    source = (REPO_ROOT / "src/trend/spec.py").read_text()
    assert '_cfg_value(cfg, "' + 'jobs"' not in source
