from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

DOC_PATH = Path("docs/phase-3/MonteCarlo.md")


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```ya?ml\n(.*?)\n```", text, flags=re.DOTALL)


def _cost_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in _yaml_blocks(text):
        if not re.search(r"(^|\n)\s+costs:|(^|\n)costs:", block):
            continue
        cleaned = "\n".join(line for line in block.splitlines() if line.strip() not in {"..."})
        parsed = yaml.safe_load(cleaned)
        if not isinstance(parsed, dict):
            continue
        costs = parsed.get("costs")
        scenario = parsed.get("scenario")
        if costs is None and isinstance(scenario, dict):
            costs = scenario.get("costs")
        assert isinstance(costs, dict), f"costs block should parse as mapping:\n{block}"
        blocks.append(costs)
    return blocks


def _walk_keys(value: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            keys.add(path)
            keys.update(_walk_keys(child, path))
    return keys


def test_monte_carlo_cost_examples_use_canonical_shape() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    cost_blocks = _cost_blocks(text)

    assert len(cost_blocks) == 2
    for costs in cost_blocks:
        assert costs["kind"] == "regime_stochastic"
        assert costs["default_regime"] == "calm"
        forbidden_paths = {
            path
            for path in _walk_keys(costs)
            if path[-1] in {"regimes", "distribution", "std", "default"}
        }
        assert not forbidden_paths

        regime_names = [name for name in costs if name not in {"kind", "default_regime"}]
        assert regime_names
        for regime in regime_names:
            regime_block = costs[regime]
            assert isinstance(regime_block, dict)
            assert "trade_cost_bps" in regime_block
            distribution = regime_block["trade_cost_bps"]
            assert isinstance(distribution, dict)
            assert {"kind", "mean", "sigma"} <= set(distribution)
            assert "dist" not in distribution


def test_documentation_states_that_legacy_cost_shapes_are_rejected() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "\n## Rejected legacy cost shapes\n" in text
    assert "rejects nested `costs.regimes`" in text
    assert "`distribution` and `dist` aliases" in text
    assert "numeric shorthand" in text


def _loaded_model(spec, tmp_path, *, nested=False):
    from tests.monte_carlo.test_runner import _base_config, _price_history
    from trend_analysis.monte_carlo.registry import load_scenario_from_path
    from trend_analysis.monte_carlo.runner import MonteCarloRunner

    payload = {
        "scenario": {"name": "published-model"},
        "base_config": str(Path("config/defaults.yml").resolve()),
        "monte_carlo": {"mode": "two_layer", "n_paths": 1, "horizon_years": 1, "frequency": "M"},
    }
    (payload["scenario"] if nested else payload)["return_model"] = spec
    path = tmp_path / "scenario.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    scenario = load_scenario_from_path(path)
    history = _price_history()
    history["SPX"] = history["AssetA"]
    runner = MonteCarloRunner(scenario, base_config=_base_config(), price_history=history)
    return scenario, runner._build_price_model()


def test_published_return_model_yaml_constructs_real_models(tmp_path):
    from trend_analysis.monte_carlo.models import (
        RegimeConditionedBootstrapModel,
        StationaryBootstrapModel,
    )

    specs = []
    for block in _yaml_blocks(DOC_PATH.read_text()):
        if not re.search(r"(^|\n)\s*return_model:", block):
            continue
        parsed = yaml.safe_load(
            "\n".join(line for line in block.splitlines() if line.strip() != "...")
        )
        spec = parsed.get("return_model") or parsed.get("scenario", {}).get("return_model")
        assert isinstance(spec, dict)
        specs.append(spec)
    assert len(specs) == 3
    for spec in specs:
        scenario, model = _loaded_model(spec, tmp_path)
        assert model.mean_block_len == spec["mean_block_len"]
        assert model.calibration_window == spec.get("calibration_window")
        assert set(scenario.return_model) == {"kind", "params"}
        if "regime" in spec:
            assert isinstance(model, RegimeConditionedBootstrapModel)
            assert model.regime_proxy_column == spec["regime"]["proxy_column"]
            assert model.threshold_percentile == spec["regime"]["threshold_percentile"]
            assert model._labels is not None and model._transition is not None
            assert model._regime_buckets is not None
        else:
            assert isinstance(model, StationaryBootstrapModel)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "kind", ["stationary_bootstrap", "bootstrap", "regime_conditioned_bootstrap"]
)
def test_nondefault_return_model_parameters_survive_loading(kind, nested, tmp_path):
    spec = {"kind": kind, "mean_block_len": 12, "calibration_window": 24}
    if kind == "regime_conditioned_bootstrap":
        spec["regime"] = {
            "enabled": True,
            "kind": "proxy_vol_threshold",
            "proxy_column": "SPX",
            "lookback": 7,
        }
    scenario, model = _loaded_model(spec, tmp_path, nested=nested)
    assert model.mean_block_len == 12
    assert model.calibration_window == 24
    assert scenario.return_model["params"]["mean_block_len"] == 12
    if kind == "regime_conditioned_bootstrap":
        assert model.regime_proxy_column == "SPX"
        assert model.lookback == 7
        assert model._labels is not None and model._transition is not None
        assert model._regime_buckets is not None


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "hmm"},
        {"kind": "stationary_bootstrap", "unknown": 1},
        {"kind": "stationary_bootstrap", "params": {"unknown": 1}},
        {"kind": "stationary_bootstrap", "mean_block_len": 12, "params": {"mean_block_len": 6}},
        {"kind": "stationary_bootstrap", "params": {"mean_block_len": 12, "block_size": 12}},
        {"kind": "stationary_bootstrap", "regime": {"enabled": True}},
        {"kind": "regime_conditioned_bootstrap", "regime": {"enabled": False}},
        {"kind": "regime_conditioned_bootstrap", "regime": {"kind": "hmm"}},
        {"kind": "stationary_bootstrap", "params": None},
        {"kind": "stationary_bootstrap", "regime_proxy_column": "SPX"},
        {"kind": "regime_bootstrap", "regime": {"enabled": "false"}},
    ],
)
def test_return_model_rejects_unknown_or_conflicting_settings(spec, tmp_path):
    with pytest.raises(ValueError, match="return_model"):
        _loaded_model(spec, tmp_path)


@pytest.mark.parametrize("nested", [False, True])
def test_disabled_regime_preserves_dormant_settings_as_a_toggle(tmp_path, nested):
    from trend_analysis.monte_carlo.models import (
        StationaryBootstrapModel,
        RegimeConditionedBootstrapModel,
    )

    spec = {
        "kind": "stationary_bootstrap",
        "mean_block_len": 12,
        "regime": {
            "enabled": False,
            "kind": "proxy_vol_threshold",
            "proxy_column": "SPX",
            "threshold_percentile": 75,
            "lookback": 7,
        },
    }
    scenario, model = _loaded_model(spec, tmp_path, nested=nested)
    assert isinstance(model, StationaryBootstrapModel)
    assert not isinstance(model, RegimeConditionedBootstrapModel)
    assert scenario.return_model == {
        "kind": "stationary_bootstrap",
        "params": {"mean_block_len": 12},
    }
