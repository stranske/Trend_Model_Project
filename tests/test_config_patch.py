"""Unit tests for config patch schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trend_analysis.config.patch import ConfigPatch, PatchOperation, RiskFlag


def test_patch_operation_accepts_dotpath_set() -> None:
    op = PatchOperation(op="set", path="vol_adjust.target_vol", value=0.12)
    assert op.op == "set"
    assert op.path == "vol_adjust.target_vol"
    assert op.value == 0.12


def test_patch_operation_accepts_json_pointer() -> None:
    op = PatchOperation(op="remove", path="/portfolio/constraints/max_weight")
    assert op.path == "/portfolio/constraints/max_weight"


@pytest.mark.parametrize(
    ("op", "path", "value"),
    [
        ("set", "", 1),
        ("set", "portfolio..constraints", 1),
        ("set", "/portfolio//constraints", 1),
        ("set", "/portfolio/constraints/", 1),
    ],
)
def test_patch_operation_rejects_invalid_paths(op: str, path: str, value: int) -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op=op, path=path, value=value)


def test_patch_operation_requires_value_for_set() -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op="set", path="portfolio.max_turnover")


def test_patch_operation_rejects_remove_with_value() -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op="remove", path="portfolio.max_turnover", value=1.0)


def test_config_patch_detects_risk_flags() -> None:
    operations = [
        PatchOperation(op="remove", path="portfolio.constraints.max_weight"),
        PatchOperation(op="set", path="vol_adjust.target_vol", value=0.2),
        PatchOperation(op="set", path="robustness.condition_check.enabled", value=False),
        PatchOperation(op="merge", path="portfolio", value={"foo": "bar"}),
    ]
    patch = ConfigPatch(operations=operations, summary="Adjust risk posture")
    assert RiskFlag.REMOVES_CONSTRAINT in patch.risk_flags
    assert RiskFlag.INCREASES_LEVERAGE in patch.risk_flags
    assert RiskFlag.REMOVES_VALIDATION in patch.risk_flags
    assert RiskFlag.BROAD_SCOPE in patch.risk_flags


def test_config_patch_empty_operations_allowed() -> None:
    patch = ConfigPatch(operations=[], summary="No changes")
    assert patch.operations == []
    assert patch.risk_flags == []


def test_config_patch_schema_export() -> None:
    schema = ConfigPatch.json_schema()
    assert "properties" in schema
    assert "operations" in schema["properties"]
