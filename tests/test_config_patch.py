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


def test_patch_operation_accepts_json_pointer_escapes() -> None:
    op = PatchOperation(op="set", path="/portfolio/~0tilde/~1slash", value=True)
    assert op.path == "/portfolio/~0tilde/~1slash"


@pytest.mark.parametrize(
    ("op", "path", "value"),
    [
        ("set", "", 1),
        ("set", "portfolio..constraints", 1),
        ("set", "/portfolio//constraints", 1),
        ("set", "/portfolio/constraints/", 1),
        ("set", "/portfolio/~2/constraints", 1),
        ("set", "/portfolio/invalid~", 1),
    ],
)
def test_patch_operation_rejects_invalid_paths(op: str, path: str, value: int) -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op=op, path=path, value=value)


def test_patch_operation_invalid_path_error_message() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PatchOperation(op="set", path="portfolio..constraints", value=1)
    assert "path must be a dotpath or JSONPointer" in str(excinfo.value)


def test_patch_operation_rejects_unknown_op() -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op="replace", path="portfolio.constraints", value=1)


def test_patch_operation_requires_value_for_set() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PatchOperation(op="set", path="portfolio.max_turnover")
    assert "value is required for op 'set'" in str(excinfo.value)


def test_patch_operation_accepts_set_with_none_value() -> None:
    op = PatchOperation(op="set", path="portfolio.max_turnover", value=None)
    assert op.value is None


@pytest.mark.parametrize("op", ["append", "merge"])
def test_patch_operation_requires_value_for_append_merge(op: str) -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op=op, path="portfolio.constraints")


def test_patch_operation_accepts_append_with_value() -> None:
    op = PatchOperation(
        op="append", path="portfolio.constraints.allowed_assets", value="ABC"
    )
    assert op.op == "append"
    assert op.value == "ABC"


def test_patch_operation_accepts_merge_with_value() -> None:
    op = PatchOperation(op="merge", path="portfolio", value={"foo": "bar"})
    assert op.op == "merge"
    assert op.value == {"foo": "bar"}


def test_patch_operation_rejects_merge_with_non_dict_value() -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op="merge", path="portfolio", value=["foo"])


def test_patch_operation_rejects_remove_with_value() -> None:
    with pytest.raises(ValidationError):
        PatchOperation(op="remove", path="portfolio.max_turnover", value=1.0)


def test_patch_operation_accepts_remove_without_value() -> None:
    op = PatchOperation(op="remove", path="portfolio.max_turnover")
    assert op.value is None


def test_config_patch_detects_risk_flags() -> None:
    operations = [
        PatchOperation(op="remove", path="portfolio.constraints.max_weight"),
        PatchOperation(op="set", path="vol_adjust.target_vol", value=0.2),
        PatchOperation(
            op="set", path="robustness.condition_check.enabled", value=False
        ),
        PatchOperation(op="merge", path="portfolio", value={"foo": "bar"}),
    ]
    patch = ConfigPatch(operations=operations, summary="Adjust risk posture")
    assert RiskFlag.REMOVES_CONSTRAINT in patch.risk_flags
    assert RiskFlag.INCREASES_LEVERAGE in patch.risk_flags
    assert RiskFlag.REMOVES_VALIDATION in patch.risk_flags
    assert RiskFlag.BROAD_SCOPE in patch.risk_flags


def test_config_patch_set_none_triggers_constraint_risk() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="set",
                path="portfolio.constraints.max_weight",
                value=None,
            )
        ],
        summary="Remove max weight constraint",
    )
    assert patch.risk_flags == [RiskFlag.REMOVES_CONSTRAINT]


def test_config_patch_empty_operations_allowed() -> None:
    patch = ConfigPatch(operations=[], summary="No changes")
    assert patch.operations == []
    assert patch.risk_flags == []


def test_config_patch_schema_export() -> None:
    schema = ConfigPatch.json_schema()
    assert "properties" in schema
    assert "operations" in schema["properties"]
    assert schema["properties"]["operations"]["description"]
    assert schema["properties"]["summary"]["description"]


def test_config_patch_accepts_append_without_broad_scope_flag() -> None:
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
    assert patch.risk_flags == []


def test_config_patch_rejects_invalid_risk_flag() -> None:
    with pytest.raises(ValidationError):
        ConfigPatch(
            operations=[],
            summary="Invalid risk flag",
            risk_flags=["NOT_A_FLAG"],
        )


def test_config_patch_requires_summary() -> None:
    with pytest.raises(ValidationError):
        ConfigPatch(operations=[])


def test_config_patch_rejects_blank_summary() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ConfigPatch(operations=[], summary="   ")
    assert "summary must be a non-empty string" in str(excinfo.value)
