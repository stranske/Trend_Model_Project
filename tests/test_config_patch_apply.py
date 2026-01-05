"""Tests for applying config patches to mappings."""

from __future__ import annotations

from trend_analysis.config.patch import ConfigPatch, PatchOperation, apply_config_patch


def test_apply_config_patch_set_creates_missing_path() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="portfolio.max_turnover", value=0.25)],
        summary="Set turnover",
    )
    updated = apply_config_patch({}, patch)
    assert updated == {"portfolio": {"max_turnover": 0.25}}


def test_apply_config_patch_set_creates_nested_path() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="set",
                path="portfolio.constraints.position_limits",
                value={"max_weight": 0.1},
            )
        ],
        summary="Set nested limit",
    )
    updated = apply_config_patch({}, patch)
    assert updated == {
        "portfolio": {"constraints": {"position_limits": {"max_weight": 0.1}}}
    }


def test_apply_config_patch_append_creates_list() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="append",
                path="portfolio.constraints.allowed_assets",
                value="ABC",
            )
        ],
        summary="Add asset",
    )
    updated = apply_config_patch({"portfolio": {"constraints": {}}}, patch)
    assert updated["portfolio"]["constraints"]["allowed_assets"] == ["ABC"]


def test_apply_config_patch_append_to_existing_list() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="append",
                path="portfolio.constraints.allowed_assets",
                value="XYZ",
            )
        ],
        summary="Add asset",
    )
    updated = apply_config_patch({"portfolio": {"constraints": {"allowed_assets": ["ABC"]}}}, patch)
    assert updated["portfolio"]["constraints"]["allowed_assets"] == ["ABC", "XYZ"]


def test_apply_config_patch_merge_deep() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="merge",
                path="portfolio",
                value={"constraints": {"max_weight": 0.05}},
            )
        ],
        summary="Merge constraints",
    )
    updated = apply_config_patch({"portfolio": {"rebalance": "M"}}, patch)
    assert updated == {"portfolio": {"rebalance": "M", "constraints": {"max_weight": 0.05}}}


def test_apply_config_patch_remove_missing_path_noop() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="remove", path="portfolio.max_turnover")],
        summary="Remove turnover",
    )
    updated = apply_config_patch({}, patch)
    assert updated == {}
