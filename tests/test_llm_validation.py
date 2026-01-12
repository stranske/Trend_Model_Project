"""Tests for LLM config patch key validation."""

from __future__ import annotations

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.llm.validation import flag_unknown_keys, validate_patch_keys


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer"},
                    "target_vol": {"type": "number"},
                },
            }
        },
    }


def _schema_with_array() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "object",
                "properties": {
                    "compute": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"window": {"type": "integer"}},
                        },
                    }
                },
            }
        },
    }


def _schema_with_array_wildcard() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "limits": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        }
                    },
                },
            }
        },
    }


def _schema_with_wildcard() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "portfolio": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "group_caps": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            }
                        },
                    }
                },
            }
        },
    }


def test_validate_patch_keys_accepts_known_paths() -> None:
    operations = [PatchOperation(op="set", path="analysis.top_n", value=12)]

    unknown = validate_patch_keys(operations, _schema())

    assert unknown == []


def test_validate_patch_keys_flags_unknown_paths() -> None:
    operations = [PatchOperation(op="set", path="analysis.unknown", value=1)]

    unknown = validate_patch_keys(operations, _schema())

    assert len(unknown) == 1
    assert unknown[0].path == "analysis.unknown"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_suggests_close_match() -> None:
    operations = [PatchOperation(op="set", path="analysis.top_nn", value=12)]

    unknown = validate_patch_keys(operations, _schema())

    assert len(unknown) == 1
    assert unknown[0].suggestion == "analysis.top_n"


def test_validate_patch_keys_flags_array_indices_for_review() -> None:
    operations = [PatchOperation(op="set", path="metrics.compute[3].window", value=12)]

    unknown = validate_patch_keys(operations, _schema_with_array())

    assert len(unknown) == 1
    assert unknown[0].path == "metrics.compute[3].window"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_json_pointer_array_indices_for_review() -> None:
    operations = [PatchOperation(op="set", path="/metrics/compute/7/window", value=12)]

    unknown = validate_patch_keys(operations, _schema_with_array())

    assert len(unknown) == 1
    assert unknown[0].path == "metrics.compute[7].window"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_wildcard_keys_for_review() -> None:
    operations = [PatchOperation(op="set", path="portfolio.constraints.group_caps.*", value=0.1)]

    unknown = validate_patch_keys(operations, _schema_with_wildcard())

    assert len(unknown) == 1
    assert unknown[0].path == "portfolio.constraints.group_caps.*"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_array_wildcard_keys_for_review() -> None:
    operations = [PatchOperation(op="set", path="rules[1].limits.*", value=0.2)]

    unknown = validate_patch_keys(operations, _schema_with_array_wildcard())

    assert len(unknown) == 1
    assert unknown[0].path == "rules[1].limits.*"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_array_index_wildcard_for_review() -> None:
    operations = [PatchOperation(op="set", path="rules[*].limits.*", value=0.2)]

    unknown = validate_patch_keys(operations, _schema_with_array_wildcard())

    assert len(unknown) == 1
    assert unknown[0].path == "rules[*].limits.*"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_unknown_array_item_key() -> None:
    operations = [PatchOperation(op="set", path="metrics.compute[2].missing_key", value=12)]

    unknown = validate_patch_keys(operations, _schema_with_array())

    assert len(unknown) == 1
    assert unknown[0].path == "metrics.compute[2].missing_key"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_unknown_json_pointer_array_item_key() -> None:
    operations = [PatchOperation(op="set", path="/metrics/compute/0/missing_key", value=12)]

    unknown = validate_patch_keys(operations, _schema_with_array())

    assert len(unknown) == 1
    assert unknown[0].path == "metrics.compute[0].missing_key"
    assert unknown[0].suggestion is None


def test_validate_patch_keys_flags_unknown_wildcard_child() -> None:
    operations = [
        PatchOperation(
            op="set",
            path="portfolio.constraints.group_caps.*.limit",
            value=0.2,
        )
    ]

    unknown = validate_patch_keys(operations, _schema_with_wildcard())

    assert len(unknown) == 1
    assert unknown[0].path == "portfolio.constraints.group_caps.*.limit"
    assert unknown[0].suggestion == "portfolio.constraints.group_caps"


def test_flag_unknown_keys_marks_review() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="analysis.unknown", value=1)],
        summary="Unknown key update.",
    )

    unknown = flag_unknown_keys(patch, _schema())

    assert patch.needs_review is True
    assert len(unknown) == 1


def test_flag_unknown_keys_skips_known_paths() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="analysis.top_n", value=12)],
        summary="Known key update.",
    )

    unknown = flag_unknown_keys(patch, _schema())

    assert patch.needs_review is False
    assert unknown == []


def test_flag_unknown_keys_marks_review_for_wildcard_child() -> None:
    patch = ConfigPatch(
        operations=[
            PatchOperation(
                op="set",
                path="portfolio.constraints.group_caps.*.limit",
                value=0.2,
            )
        ],
        summary="Wildcard child update.",
    )

    unknown = flag_unknown_keys(patch, _schema_with_wildcard())

    assert patch.needs_review is True
    assert len(unknown) == 1


def test_flag_unknown_keys_marks_review_for_array_index_wildcard() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="rules[*].limits.*", value=0.2)],
        summary="Array wildcard update.",
    )

    unknown = flag_unknown_keys(patch, _schema_with_array_wildcard())

    assert patch.needs_review is True
    assert len(unknown) == 1


def test_flag_unknown_keys_marks_review_for_wildcard_key() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="portfolio.constraints.group_caps.*", value=0.1)],
        summary="Wildcard key update.",
    )

    unknown = flag_unknown_keys(patch, _schema_with_wildcard())

    assert patch.needs_review is True
    assert len(unknown) == 1
