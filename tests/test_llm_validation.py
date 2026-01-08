"""Tests for LLM config patch key validation."""

from __future__ import annotations

from trend_analysis.config.patch import PatchOperation
from trend_analysis.llm.validation import validate_patch_keys


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
