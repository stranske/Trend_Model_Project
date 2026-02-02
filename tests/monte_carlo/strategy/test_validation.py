from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis.monte_carlo.strategy.validation import validate_strategy_pack


def test_validate_strategy_pack_accepts_hf_equity_curated() -> None:
    pack_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    errors = validate_strategy_pack(pack_path)
    assert errors == []


def test_validate_strategy_pack_rejects_non_list_curated(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_pack.yml"
    pack_path.write_text(yaml.safe_dump({"curated": "not-a-list"}), encoding="utf-8")

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated must be a list"]
