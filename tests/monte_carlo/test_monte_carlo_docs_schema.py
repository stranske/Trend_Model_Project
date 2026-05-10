from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


DOC_PATH = Path("docs/phase-3/MonteCarlo.md")


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```ya?ml\n(.*?)\n```", text, flags=re.DOTALL)


def _cost_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in _yaml_blocks(text):
        if not re.search(r"(^|\n)\s+costs:|(^|\n)costs:", block):
            continue
        cleaned = "\n".join(
            line for line in block.splitlines() if line.strip() not in {"..."}
        )
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
            assert {"dist", "mean", "sigma"} <= set(distribution)


def test_backward_compatibility_table_lists_legacy_keys_with_statuses() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "### Backward Compatibility" in text

    required_rows = {
        "`costs.regimes.<name>`": "`supported`",
        "`costs.regimes.<name>.distribution`": "`supported`",
        "`costs.<name>.distribution`": "`supported`",
        "`costs.<name>.trade_cost_bps.dist`": "`supported`",
        "`costs.<name>.trade_cost_bps.std`": "`supported`",
        "`costs.regimes.<name>: <number>`": "`supported`",
        "`costs.default`": "`deprecated`",
    }
    for legacy_key, status in required_rows.items():
        row_pattern = rf"\|\s*{re.escape(legacy_key)}\s*\|\s*{re.escape(status)}\s*\|"
        assert re.search(row_pattern, text), f"missing compatibility row for {legacy_key}"

    assert "`costs.regimes.calm.distribution.kind: fixed`" in text
    assert "`costs.calm.trade_cost_bps.kind: fixed`" in text
    assert "`costs.regimes.calm: 6`" in text
    assert "`costs.calm.trade_cost_bps.value: 6`" in text
