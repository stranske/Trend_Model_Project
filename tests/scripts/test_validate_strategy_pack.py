"""Tests for the strategy pack validation script."""

from __future__ import annotations

from pathlib import Path

from scripts import validate_strategy_pack


def test_validate_strategy_pack_main_accepts_hf_equity_curated() -> None:
    pack_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")

    exit_code = validate_strategy_pack.main([str(pack_path)])

    assert exit_code == 0
