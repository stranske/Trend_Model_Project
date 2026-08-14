"""Regression gates for the canonical missing-data configuration surface."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    "src/trend_analysis/data.py",
    "src/trend_analysis/tool_layer.py",
    "src/trend_analysis/pipeline_entrypoints.py",
    "src/trend_analysis/multi_period/engine.py",
    "src/trend_analysis/multi_period/loaders.py",
)


def test_legacy_missing_policy_keys_are_not_read() -> None:
    """Runtime code must not revive the retired missing-data config aliases."""

    forbidden = ("nan" + "_policy", "nan" + "_limit")
    offenders = {
        relative_path: [key for key in forbidden if key in (REPO_ROOT / relative_path).read_text()]
        for relative_path in RUNTIME_PATHS
    }

    assert not {path: keys for path, keys in offenders.items() if keys}
