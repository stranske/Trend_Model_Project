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


def test_validate_patch_keys_accepts_array_indices() -> None:
    operations = [PatchOperation(op="set", path="metrics.compute[3].window", value=12)]

    unknown = validate_patch_keys(operations, _schema_with_array())

    assert unknown == []


def test_validate_patch_keys_accepts_wildcard_keys() -> None:
    operations = [
        PatchOperation(op="set", path="portfolio.constraints.group_caps.*", value=0.1)
    ]

    unknown = validate_patch_keys(operations, _schema_with_wildcard())

    assert unknown == []


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
