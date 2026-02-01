from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.monte_carlo import load_scenario


def _write_registry(tmp_path: Path, name: str, filename: str) -> Path:
    registry = tmp_path / "index.yml"
    registry.write_text(
        f"scenarios:\n  - name: {name}\n    path: {filename}\n",
        encoding="utf-8",
    )
    return registry


def _write_scenario(tmp_path: Path, name: str, extra_payload: str) -> Path:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / f"{name}.yml"
    scenario_path.write_text(
        "scenario:\n"
        f"  name: {name}\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        f"{extra_payload}",
        encoding="utf-8",
    )
    return scenario_path


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("return_model", "return_model: null\n"),
        ("folds", "folds: null\n"),
    ],
)
def test_parser_rejects_null_optional_mappings(
    tmp_path: Path, field: str, payload: str
) -> None:
    name = f"null_{field}"
    _write_scenario(tmp_path, name, payload)
    registry = _write_registry(tmp_path, name, f"{name}.yml")

    with pytest.raises(
        ValueError,
        match=rf"Scenario config '{field}' must be a mapping \(null provided\)",
    ):
        load_scenario(name, registry_path=registry)


def test_parser_prefers_folds_error_when_both_null(tmp_path: Path) -> None:
    name = "null_folds_and_return_model"
    _write_scenario(tmp_path, name, "folds: null\nreturn_model: null\n")
    registry = _write_registry(tmp_path, name, f"{name}.yml")

    with pytest.raises(
        ValueError, match=r"Scenario config 'folds' must be a mapping \(null provided\)"
    ):
        load_scenario(name, registry_path=registry)
