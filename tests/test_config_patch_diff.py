"""Tests for config patch diffing helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config.patch import (
    ConfigPatch,
    PatchOperation,
    apply_and_diff,
    apply_config_patch,
    diff_configs,
)


def test_diff_configs_unified_format() -> None:
    old = {"portfolio": {"max_turnover": 0.2}}
    new = {"portfolio": {"max_turnover": 0.25}}
    diff = diff_configs(old, new)
    assert diff.startswith("--- before\n+++ after\n")
    assert "@@" in diff
    assert "-  max_turnover: 0.2" in diff
    assert "+  max_turnover: 0.25" in diff


def test_apply_and_diff_returns_updated_config_and_diff(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("portfolio:\n  max_turnover: 0.2\n", encoding="utf-8")
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="portfolio.max_turnover", value=0.3)],
        summary="Increase turnover",
    )
    updated, diff = apply_and_diff(config_path, patch)
    assert updated["portfolio"]["max_turnover"] == 0.3
    assert "+  max_turnover: 0.3" in diff


def test_apply_config_patch_invalid_path_suggests_close_match() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="portfolio.max_turnver", value=0.25)],
        summary="Set turnover",
    )
    with pytest.raises(ValueError) as excinfo:
        apply_config_patch({"portfolio": {"max_turnover": 0.2}}, patch)
    message = str(excinfo.value)
    assert "Invalid path 'portfolio.max_turnver'" in message
    assert "portfolio.max_turnover" in message


def test_apply_config_patch_type_mismatch_raises_clear_error() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(op="append", path="portfolio.constraints.allowed_assets", value="ABC")
        ],
        summary="Append asset",
    )
    with pytest.raises(ValueError) as excinfo:
        apply_config_patch({"portfolio": {"constraints": {"allowed_assets": "ABC"}}}, patch)
    assert "append requires a list" in str(excinfo.value)
