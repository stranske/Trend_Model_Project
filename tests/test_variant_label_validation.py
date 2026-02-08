"""Tests for deterministic variant label validation."""

from __future__ import annotations

import pytest

from trend_analysis.llm.chain import ConfigPatchVariants


def _patch_payload(summary: str) -> dict[str, object]:
    return {
        "operations": [],
        "risk_flags": [],
        "summary": summary,
    }


def test_variant_label_case_collision_raises() -> None:
    payload = {
        "variants": [
            {"label": "Baseline", "patch": _patch_payload("A")},
            {"label": "baseline", "patch": _patch_payload("B")},
            {"label": "Aggressive", "patch": _patch_payload("C")},
        ]
    }

    with pytest.raises(ValueError) as excinfo:
        ConfigPatchVariants.model_validate(payload)

    assert "variants must have unique labels (case-insensitive): baseline" in str(excinfo.value)
