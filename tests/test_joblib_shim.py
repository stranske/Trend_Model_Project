"""Unit tests for the lightweight joblib shim helpers."""

from __future__ import annotations

from pathlib import Path

from trend_analysis.util.joblib_shim import dump, load


def test_joblib_shim_roundtrip(tmp_path: Path) -> None:
    payload = {"a": [1, 2, 3], "b": {"nested": True}}
    target = tmp_path / "cache.joblib"

    dump(payload, target)
    assert target.exists()

    restored = load(target)
    assert restored == payload
